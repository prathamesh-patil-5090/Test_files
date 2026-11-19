"""
Invoice-approval guardrails for a finance AI agent.

Idea: before an agent may approve or pay a vendor bill, a deterministic
policy layer checks the tool call against the ledger. The model can propose
an action; it cannot bypass the gates.

Rules checked:
  1. Invoice amount must match the purchase order, within a set tolerance.
  2. The bill must not be a duplicate of an already open or paid bill
     (same vendor + invoice number, or same vendor + amount + invoice date).
  3. Amounts over a configured threshold need a prior human approval for
     the same bill and the same amount.

This is a regression suite, not a Peakflo integration. It is meant to sit
in front of agent tools such as approve_bill and pay_bill.

Run:  pytest -v test_invoice_approval_guardrails.py
"""
from dataclasses import dataclass, field
from typing import Optional

import pytest


# --- Ledger objects ----------------------------------------------------------
@dataclass(frozen=True)
class Invoice:
    invoice_id: str
    vendor_id: str
    invoice_number: str
    amount: int  # minor units, e.g. cents
    invoice_date: str  # ISO date
    po_id: str


@dataclass(frozen=True)
class PurchaseOrder:
    po_id: str
    vendor_id: str
    amount: int


@dataclass(frozen=True)
class ExistingBill:
    invoice_id: str
    vendor_id: str
    invoice_number: str
    amount: int
    invoice_date: str
    status: str  # open | paid | cancelled


@dataclass
class ToolCall:
    name: str
    args: dict = field(default_factory=dict)


@dataclass
class Trace:
    invoice: Invoice
    po: PurchaseOrder
    ledger: list
    calls: list


# --- Policy -----------------------------------------------------------------
TOLERANCE_CENTS = 100  # allow a $1 rounding / tax-cent difference
HUMAN_APPROVAL_THRESHOLD = 50_000_00  # $50,000.00


class PolicyDenied(AssertionError):
    pass


def _normalize_number(value: str) -> str:
    return value.strip().lower()


def find_duplicate(invoice: Invoice, ledger: list) -> Optional[ExistingBill]:
    for bill in ledger:
        if bill.invoice_id == invoice.invoice_id or bill.status == "cancelled":
            continue
        same_vendor = bill.vendor_id == invoice.vendor_id
        same_number = _normalize_number(bill.invoice_number) == _normalize_number(
            invoice.invoice_number
        )
        same_amount_and_date = (
            bill.amount == invoice.amount and bill.invoice_date == invoice.invoice_date
        )
        if same_vendor and (same_number or same_amount_and_date):
            return bill
    return None


def po_amount_matches(invoice: Invoice, po: PurchaseOrder, tolerance=TOLERANCE_CENTS) -> bool:
    if invoice.po_id != po.po_id or invoice.vendor_id != po.vendor_id:
        return False
    return abs(invoice.amount - po.amount) <= tolerance


def human_approval_granted(invoice: Invoice, calls_before: list) -> bool:
    for call in reversed(calls_before):
        if call.name != "request_human_approval":
            continue
        return (
            call.args.get("invoice_id") == invoice.invoice_id
            and call.args.get("amount") == invoice.amount
            and call.args.get("status") == "granted"
        )
    return False


def check_action(trace: Trace, call_index: int):
    """Raise PolicyDenied if the approve/pay call at call_index is illegal."""
    call = trace.calls[call_index]
    if call.name not in {"approve_bill", "pay_bill"}:
        return

    invoice, po = trace.invoice, trace.po
    if call.args.get("invoice_id") != invoice.invoice_id:
        raise PolicyDenied("tool call invoice_id does not match the bill in context")
    if call.args.get("amount") != invoice.amount:
        raise PolicyDenied("tool call amount does not match the bill amount")

    if not po_amount_matches(invoice, po):
        raise PolicyDenied(
            f"invoice amount {invoice.amount} does not match PO {po.amount} "
            f"within tolerance {TOLERANCE_CENTS}"
        )

    duplicate = find_duplicate(invoice, trace.ledger)
    if duplicate:
        raise PolicyDenied(
            f"duplicate of {duplicate.invoice_id} "
            f"({duplicate.invoice_number}, {duplicate.status})"
        )

    if invoice.amount > HUMAN_APPROVAL_THRESHOLD:
        prior = trace.calls[:call_index]
        if not human_approval_granted(invoice, prior):
            raise PolicyDenied(
                f"amount {invoice.amount} is over threshold "
                f"{HUMAN_APPROVAL_THRESHOLD} and has no granted human approval"
            )


def evaluate(trace: Trace):
    for i, call in enumerate(trace.calls):
        check_action(trace, i)


# --- Fixtures ----------------------------------------------------------------
VENDOR = "V-ACME"
PO = PurchaseOrder(po_id="PO-1001", vendor_id=VENDOR, amount=12_500_00)

SMALL_INVOICE = Invoice(
    invoice_id="INV-new",
    vendor_id=VENDOR,
    invoice_number="ACM-7781",
    amount=12_500_00,
    invoice_date="2026-11-10",
    po_id="PO-1001",
)

LARGE_INVOICE = Invoice(
    invoice_id="INV-large",
    vendor_id=VENDOR,
    invoice_number="ACM-9001",
    amount=75_000_00,
    invoice_date="2026-11-12",
    po_id="PO-1001-LARGE",
)

LARGE_PO = PurchaseOrder(po_id="PO-1001-LARGE", vendor_id=VENDOR, amount=75_000_00)

EMPTY_LEDGER: list = []


def _trace(invoice, po, ledger, calls):
    return Trace(invoice=invoice, po=po, ledger=ledger, calls=calls)


GOOD_TRACES = {
    "happy_path_under_threshold": _trace(
        SMALL_INVOICE,
        PO,
        EMPTY_LEDGER,
        [
            ToolCall("match_po", {"po_id": "PO-1001"}),
            ToolCall("approve_bill", {"invoice_id": "INV-new", "amount": 12_500_00}),
            ToolCall("pay_bill", {"invoice_id": "INV-new", "amount": 12_500_00}),
        ],
    ),
    "large_bill_with_human_approval": _trace(
        LARGE_INVOICE,
        LARGE_PO,
        EMPTY_LEDGER,
        [
            ToolCall("match_po", {"po_id": "PO-1001-LARGE"}),
            ToolCall(
                "request_human_approval",
                {
                    "invoice_id": "INV-large",
                    "amount": 75_000_00,
                    "status": "granted",
                },
            ),
            ToolCall("approve_bill", {"invoice_id": "INV-large", "amount": 75_000_00}),
            ToolCall("pay_bill", {"invoice_id": "INV-large", "amount": 75_000_00}),
        ],
    ),
    "tolerance_of_one_dollar_passes": _trace(
        Invoice(
            invoice_id="INV-tax",
            vendor_id=VENDOR,
            invoice_number="ACM-7782",
            amount=12_500_75,
            invoice_date="2026-11-10",
            po_id="PO-1001",
        ),
        PO,
        EMPTY_LEDGER,
        [
            ToolCall("approve_bill", {"invoice_id": "INV-tax", "amount": 12_500_75}),
        ],
    ),
    "cancelled_bill_is_not_a_duplicate": _trace(
        SMALL_INVOICE,
        PO,
        [
            ExistingBill(
                invoice_id="INV-old",
                vendor_id=VENDOR,
                invoice_number="ACM-7781",
                amount=12_500_00,
                invoice_date="2026-11-10",
                status="cancelled",
            )
        ],
        [
            ToolCall("approve_bill", {"invoice_id": "INV-new", "amount": 12_500_00}),
        ],
    ),
}

BAD_TRACES = {
    "amount_does_not_match_po": _trace(
        Invoice(
            invoice_id="INV-mismatch",
            vendor_id=VENDOR,
            invoice_number="ACM-7783",
            amount=18_000_00,
            invoice_date="2026-11-10",
            po_id="PO-1001",
        ),
        PO,
        EMPTY_LEDGER,
        [
            ToolCall("approve_bill", {"invoice_id": "INV-mismatch", "amount": 18_000_00}),
        ],
    ),
    "duplicate_invoice_number": _trace(
        SMALL_INVOICE,
        PO,
        [
            ExistingBill(
                invoice_id="INV-paid",
                vendor_id=VENDOR,
                invoice_number="acm-7781",  # case-insensitive
                amount=9_000_00,
                invoice_date="2026-08-01",
                status="paid",
            )
        ],
        [
            ToolCall("pay_bill", {"invoice_id": "INV-new", "amount": 12_500_00}),
        ],
    ),
    "near_duplicate_same_vendor_amount_date": _trace(
        SMALL_INVOICE,
        PO,
        [
            ExistingBill(
                invoice_id="INV-resubmitted",
                vendor_id=VENDOR,
                invoice_number="ACM-7781-B",  # different number, same vendor/amount/date
                amount=12_500_00,
                invoice_date="2026-11-10",
                status="open",
            )
        ],
        [
            ToolCall("approve_bill", {"invoice_id": "INV-new", "amount": 12_500_00}),
        ],
    ),
    "pays_over_threshold_without_approval": _trace(
        LARGE_INVOICE,
        LARGE_PO,
        EMPTY_LEDGER,
        [
            ToolCall("pay_bill", {"invoice_id": "INV-large", "amount": 75_000_00}),
        ],
    ),
    "approval_granted_for_a_different_amount": _trace(
        LARGE_INVOICE,
        LARGE_PO,
        EMPTY_LEDGER,
        [
            ToolCall(
                "request_human_approval",
                {
                    "invoice_id": "INV-large",
                    "amount": 40_000_00,  # not the bill amount
                    "status": "granted",
                },
            ),
            ToolCall("pay_bill", {"invoice_id": "INV-large", "amount": 75_000_00}),
        ],
    ),
    "approval_was_denied": _trace(
        LARGE_INVOICE,
        LARGE_PO,
        EMPTY_LEDGER,
        [
            ToolCall(
                "request_human_approval",
                {
                    "invoice_id": "INV-large",
                    "amount": 75_000_00,
                    "status": "denied",
                },
            ),
            ToolCall("approve_bill", {"invoice_id": "INV-large", "amount": 75_000_00}),
        ],
    ),
    "agent_pays_a_different_amount_than_the_bill": _trace(
        SMALL_INVOICE,
        PO,
        EMPTY_LEDGER,
        [
            ToolCall("pay_bill", {"invoice_id": "INV-new", "amount": 20_000_00}),
        ],
    ),
}


@pytest.mark.parametrize("name", GOOD_TRACES)
def test_good_traces_are_allowed(name):
    evaluate(GOOD_TRACES[name])


@pytest.mark.parametrize("name", BAD_TRACES)
def test_bad_traces_are_blocked(name):
    with pytest.raises(PolicyDenied):
        evaluate(BAD_TRACES[name])


def test_duplicate_rule_is_case_insensitive():
    hit = find_duplicate(
        SMALL_INVOICE,
        [
            ExistingBill(
                invoice_id="INV-paid",
                vendor_id=VENDOR,
                invoice_number="ACM-7781",
                amount=1,
                invoice_date="2020-01-01",
                status="paid",
            )
        ],
    )
    assert hit is not None
    assert hit.invoice_id == "INV-paid"
