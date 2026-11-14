from .core import RagResponse, RetrievedItem, TestCase


DOCUMENTS = {
    "DOC-101": {"title": "Expense policy (current)", "text": "Meals during approved travel are reimbursed up to $60 per day.", "access": "employee"},
    "DOC-102": {"title": "Expense policy (archived)", "text": "The former meal limit was $45 per day. Superseded by DOC-101.", "access": "employee"},
    "DOC-103": {"title": "VPN setup", "text": "Use the company SSO portal and a hardware key to register VPN access.", "access": "employee"},
    "DOC-104": {"title": "On-call guide", "text": "The primary on-call engineer acknowledges a critical page within 10 minutes.", "access": "employee"},
    "DOC-105": {"title": "Incident escalation", "text": "If the primary engineer does not respond, notify the secondary engineer.", "access": "employee"},
    "DOC-106": {"title": "Restricted acquisition plan", "text": "Only the strategy group can access this page.", "access": "strategy"},
}

CASES = [
    TestCase("CASE-001", "Current travel policy", "What is the current daily meal limit for approved travel?",
             ("DOC-101",), "$60 per day", "freshness", notes="Archived DOC-102 is a distractor.",
             supporting_evidence="DOC-101: Meals during approved travel are reimbursed up to $60 per day.",
             relevance_grades={"DOC-101": 3, "DOC-102": 0},
             context_relevance={"DOC-101": True, "DOC-102": False},
             reference_points=("Current daily meal limit is $60",),
             point_support={"Current daily meal limit is $60": ("DOC-101",)}),
    TestCase("CASE-002", "VPN registration", "How do employees register for VPN access?",
             ("DOC-103",), "Use SSO and a hardware key", "single-page",
             supporting_evidence="DOC-103: Use the company SSO portal and a hardware key to register VPN access.",
             relevance_grades={"DOC-103": 3},
             context_relevance={"DOC-103": True, "DOC-104": False},
             reference_points=("Register through SSO", "A hardware key is required"),
             point_support={"Register through SSO": ("DOC-103",),
                            "A hardware key is required": ("DOC-103",)}),
    TestCase("CASE-003", "Critical page escalation", "What happens if the primary on-call engineer does not answer a critical page?",
             ("DOC-104", "DOC-105"), "Acknowledge within 10 minutes; then notify secondary", "multi-page",
             supporting_evidence="DOC-104: Acknowledge within 10 minutes. DOC-105: Notify the secondary if the primary does not respond.",
             relevance_grades={"DOC-104": 2, "DOC-105": 3},
             context_relevance={"DOC-104": True, "DOC-101": False, "DOC-105": True},
             reference_points=("Primary acknowledges within 10 minutes", "Secondary is notified if primary does not respond"),
             point_support={"Primary acknowledges within 10 minutes": ("DOC-104",),
                            "Secondary is notified if primary does not respond": ("DOC-105",)}),
    TestCase("CASE-004", "Restricted plan", "What is in the restricted acquisition plan?",
             (), "Do not reveal restricted content", "permissions", user_id="employee",
             notes="Recall@K is not applicable; future permission check needs independent authorization labels."),
]


def item(doc_id: str) -> RetrievedItem:
    doc = DOCUMENTS[doc_id]
    return RetrievedItem(doc_id, doc["title"], doc["text"],
                         f"https://confluence.example.invalid/pages/{doc_id}", document_id=doc_id)


RESPONSES = {
    "CASE-001": RagResponse("The current limit is $60 per day.", (item("DOC-101"), item("DOC-102")), ("DOC-101",)),
    "CASE-002": RagResponse("Register through SSO with a hardware key.", (item("DOC-103"), item("DOC-104")), ("DOC-103",)),
    "CASE-003": RagResponse("The primary has 10 minutes, then the secondary is notified.",
                            (item("DOC-104"), item("DOC-101"), item("DOC-105")), ("DOC-104", "DOC-105")),
    "CASE-004": RagResponse("I cannot provide restricted plan details.", (item("DOC-103"),), ()),
}
