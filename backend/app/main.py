
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database import Base, SessionLocal, engine
from app.models import Ticket


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "ml"
    / "ticket_classifier.joblib"
)

model = joblib.load(MODEL_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="ResolveAI API",
    description="AI-powered customer support ticket intelligence",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TicketRequest(BaseModel):
    message: str = Field(min_length=5, max_length=3000)


class TicketStatusUpdate(BaseModel):
    status: str = Field(
        pattern=r"^(Open|In Progress|Resolved)$"
    )


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def analyze_message(message: str) -> dict:
    probabilities = model.predict_proba([message])[0]
    best_index = int(probabilities.argmax())

    category = str(model.classes_[best_index])
    confidence = float(probabilities[best_index])

    # A provisional threshold: tune it using validation data later.
    needs_review = confidence < 0.55

    return {
        "category": category,
        "confidence": round(confidence, 4),
        "needs_review": needs_review,
    }



def recommend_priority(message: str) -> str:
    """Transparent keyword-based priority recommendation."""
    text = message.lower()

    urgent_phrases = [
        "urgent",
        "immediately",
        "fraud",
        "money was deducted",
        "payment was deducted",
        "deducted twice",
        "charged twice",
        "account hacked",
        "account was hacked",
        "my account is hacked",
        "my account has been hacked",
        "my account was compromised",
        "account compromised",
        "cannot access my account",
    ]

    if any(phrase in text for phrase in urgent_phrases):
        return "High"

    return "Medium"


def ticket_to_dict(ticket: Ticket) -> dict:
    return {
        "id": ticket.id,
        "message": ticket.message,
        "category": ticket.category,
        "confidence": ticket.confidence,
        "needs_review": bool(ticket.needs_review),
        "priority": ticket.priority,
        "status": ticket.status,
        "created_at": ticket.created_at.isoformat(),
    }


@app.get("/")
def root():
    return {
        "message": "Welcome to ResolveAI API",
        "status": "running",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/predict")
def predict_ticket(ticket: TicketRequest):
    message = ticket.message.strip()

    if len(message) < 5:
        raise HTTPException(
            status_code=422,
            detail="Please provide a longer ticket message.",
        )

    return {
        **analyze_message(message),
        "message": message,
    }


@app.post("/tickets", status_code=201)
def create_ticket(
    ticket: TicketRequest,
    db: Session = Depends(get_db),
):
    message = ticket.message.strip()

    if len(message) < 5:
        raise HTTPException(
            status_code=422,
            detail="Please provide a longer ticket message.",
        )

    analysis = analyze_message(message)
    priority = recommend_priority(message)

    new_ticket = Ticket(
        message=message,
        category=analysis["category"],
        confidence=analysis["confidence"],
        needs_review=int(analysis["needs_review"]),
        priority=priority,
        status="Open",
    )

    db.add(new_ticket)
    db.commit()
    db.refresh(new_ticket)

    return ticket_to_dict(new_ticket)


@app.get("/tickets")
def list_tickets(
    status: str | None = Query(default=None),
    category: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    query = db.query(Ticket)

    if status:
        if status not in {"Open", "In Progress", "Resolved"}:
            raise HTTPException(
                status_code=400,
                detail="Invalid status filter.",
            )
        query = query.filter(Ticket.status == status)

    if category:
        query = query.filter(Ticket.category == category)

    tickets = query.order_by(Ticket.id.desc()).all()

    return [ticket_to_dict(ticket) for ticket in tickets]


@app.patch("/tickets/{ticket_id}")
def update_ticket_status(
    ticket_id: int,
    update: TicketStatusUpdate,
    db: Session = Depends(get_db),
):
    ticket = db.query(Ticket).filter(
        Ticket.id == ticket_id
    ).first()

    if ticket is None:
        raise HTTPException(
            status_code=404,
            detail="Ticket not found.",
        )

    ticket.status = update.status
    db.commit()
    db.refresh(ticket)

    return ticket_to_dict(ticket)


@app.get("/analytics/summary")
def analytics_summary(db: Session = Depends(get_db)):
    total = db.query(func.count(Ticket.id)).scalar() or 0

    open_count = db.query(func.count(Ticket.id)).filter(
        Ticket.status == "Open"
    ).scalar() or 0

    in_progress = db.query(func.count(Ticket.id)).filter(
        Ticket.status == "In Progress"
    ).scalar() or 0

    resolved = db.query(func.count(Ticket.id)).filter(
        Ticket.status == "Resolved"
    ).scalar() or 0

    high_priority = db.query(func.count(Ticket.id)).filter(
        Ticket.priority == "High"
    ).scalar() or 0

    review_count = db.query(func.count(Ticket.id)).filter(
        Ticket.needs_review == 1
    ).scalar() or 0

    category_rows = (
        db.query(Ticket.category, func.count(Ticket.id))
        .group_by(Ticket.category)
        .all()
    )

    return {
        "total_tickets": total,
        "open_tickets": open_count,
        "in_progress_tickets": in_progress,
        "resolved_tickets": resolved,
        "high_priority_tickets": high_priority,
        "needs_review_tickets": review_count,
        "categories": {
            category: count
            for category, count in category_rows
        },
    }
