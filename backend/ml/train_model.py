
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

# Each message is paired with its correct category.
# These are illustrative training examples, not real customer data.
tickets = [
    ("My refund has not arrived after the cancellation", "Refund"),
    ("I am still waiting for my money back", "Refund"),
    ("Please check the status of my refund", "Refund"),
    ("The cancelled order refund is delayed", "Refund"),
    ("When will the refund reach my bank account", "Refund"),
    ("I received only part of my refund", "Refund"),
    ("My money was not returned after the return", "Refund"),
    ("The refund transaction is taking too long", "Refund"),

    ("My payment was deducted twice", "Payment"),
    ("The card payment failed during checkout", "Payment"),
    ("I was charged but the order was not placed", "Payment"),
    ("The payment gateway keeps declining my card", "Payment"),
    ("I cannot complete the online transaction", "Payment"),
    ("The wallet balance was deducted incorrectly", "Payment"),
    ("I received a duplicate payment charge", "Payment"),
    ("The transaction failed but money was debited", "Payment"),

    ("The application crashes when I open it", "Technical"),
    ("The website displays a server error", "Technical"),
    ("The mobile app freezes on the loading screen", "Technical"),
    ("The page is not loading properly", "Technical"),
    ("I get an error whenever I upload a file", "Technical"),
    ("The dashboard is broken and unresponsive", "Technical"),
    ("The application keeps closing unexpectedly", "Technical"),
    ("The website is extremely slow and unresponsive", "Technical"),

    ("I cannot log into my account", "Account"),
    ("Please help me reset my password", "Account"),
    ("How can I change my registered email address", "Account"),
    ("My account has been locked", "Account"),
    ("I forgot my username", "Account"),
    ("I need to update my account information", "Account"),
    ("I cannot verify my account details", "Account"),
    ("How do I change my login credentials", "Account"),

    ("Where is my order and tracking number", "Shipping"),
    ("My package has not been delivered", "Shipping"),
    ("The delivery is delayed by three days", "Shipping"),
    ("I received the wrong delivery address confirmation", "Shipping"),
    ("The tracking page has not updated", "Shipping"),
    ("My parcel is missing from the shipment", "Shipping"),
    ("Can I change the delivery date", "Shipping"),
    ("The courier has not arrived yet", "Shipping"),
]


# Additional synthetic examples to broaden wording coverage.
# These are for development only, not a substitute for real support data.
additional_tickets = [
    # Refund
    ("The refund is still pending in my bank", "Refund"),
    ("I returned the product but have not received my money", "Refund"),
    ("Why was my refund request rejected", "Refund"),
    ("The refund amount is different from what I paid", "Refund"),
    ("I cancelled my purchase and need the refund status", "Refund"),
    ("My refund confirmation arrived but no money was credited", "Refund"),
    ("Please investigate the missing refund for my order", "Refund"),
    ("I have been waiting two weeks for my refund", "Refund"),
    ("The refund was processed to the wrong payment method", "Refund"),
    ("I want to know when my return payment will arrive", "Refund"),
    ("My refund shows completed but the balance is unchanged", "Refund"),
    ("I was promised a refund but it has not been issued", "Refund"),

    # Payment
    ("My bank transfer failed but the balance decreased", "Payment"),
    ("The checkout says payment unsuccessful", "Payment"),
    ("I cannot add my card to make a payment", "Payment"),
    ("My UPI transaction is pending", "Payment"),
    ("The payment page shows an unexpected error", "Payment"),
    ("I was billed twice for the same order", "Payment"),
    ("The payment was declined even though I have funds", "Payment"),
    ("My invoice shows an incorrect charge", "Payment"),
    ("The transaction is pending but the amount was debited", "Payment"),
    ("I need help with a failed wallet payment", "Payment"),
    ("The card was charged but the checkout timed out", "Payment"),
    ("My payment receipt has the wrong amount", "Payment"),

    # Technical
    ("I see a blank screen after signing in", "Technical"),
    ("The app crashes when I submit the form", "Technical"),
    ("The website returns a 500 error", "Technical"),
    ("Images are not uploading on the support page", "Technical"),
    ("The application stopped responding today", "Technical"),
    ("I cannot open the settings page because it keeps loading", "Technical"),
    ("The search feature returns an error", "Technical"),
    ("The site gives me a connection timeout", "Technical"),
    ("The latest update made the app freeze", "Technical"),
    ("The dashboard data is not refreshing", "Technical"),
    ("A button does nothing when I click it", "Technical"),
    ("The application displays a network error", "Technical"),

    # Account
    ("Someone may have accessed my account without permission", "Account"),
    ("I cannot receive the password reset email", "Account"),
    ("My login code is not working", "Account"),
    ("I need to change the email linked to my profile", "Account"),
    ("My account verification keeps failing", "Account"),
    ("I think my profile has been compromised", "Account"),
    ("I am locked out after too many login attempts", "Account"),
    ("How can I enable two factor authentication", "Account"),
    ("My registered phone number is incorrect", "Account"),
    ("I cannot sign in with my usual credentials", "Account"),
    ("Please help me recover access to my profile", "Account"),
    ("I need to update my personal account details", "Account"),

    # Shipping
    ("The tracking number says shipment not found", "Shipping"),
    ("My order is marked delivered but I received nothing", "Shipping"),
    ("The courier delivered my package to another address", "Shipping"),
    ("Can I update the shipping address for my order", "Shipping"),
    ("My parcel has been stuck in transit for days", "Shipping"),
    ("The delivery agent has not contacted me", "Shipping"),
    ("I need an update on the estimated delivery date", "Shipping"),
    ("The package arrived damaged during delivery", "Shipping"),
    ("My shipment tracking has not changed since dispatch", "Shipping"),
    ("The courier attempted delivery at the wrong location", "Shipping"),
    ("I received a notification but my parcel is missing", "Shipping"),
    ("Can you check why my order has not shipped yet", "Shipping"),
]

tickets.extend(additional_tickets)



def build_model():
    """Create a fresh TF-IDF and logistic-regression pipeline."""
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                strip_accents="unicode",
            ),
        ),
        (
            "classifier",
            LogisticRegression(max_iter=1000, random_state=42),
        ),
    ])


def get_dataset():
    """Return the illustrative ticket messages and their labels."""
    texts = [message for message, category in tickets]
    labels = [category for message, category in tickets]
    return texts, labels


def train_and_evaluate():
    """Evaluate the model and save the final trained pipeline."""
    texts, labels = get_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )

    model = build_model()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"},
    )

    print("5-fold cross-validation on training data:")
    print(
        f"Accuracy: {cv_results['test_accuracy'].mean():.3f} "
        f"+/- {cv_results['test_accuracy'].std():.3f}"
    )
    print(
        f"Macro F1: {cv_results['test_macro_f1'].mean():.3f} "
        f"+/- {cv_results['test_macro_f1'].std():.3f}"
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"\\nTraining examples: {len(X_train)}")
    print(f"Test examples: {len(X_test)}")
    print("\\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("\\nConfusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_test, predictions, labels=model.classes_))
    print("Class order:", list(model.classes_))

    model_path = Path(__file__).parent / "ticket_classifier.joblib"
    joblib.dump(model, model_path)
    print(f"\\nModel saved to: {model_path}")
    return model


def predict_category(message: str, model=None) -> dict:
    """Predict a category; load the saved model when one is not supplied."""
    if model is None:
        model_path = Path(__file__).parent / "ticket_classifier.joblib"
        model = joblib.load(model_path)

    probabilities = model.predict_proba([message])[0]
    best_index = int(probabilities.argmax())
    category = str(model.classes_[best_index])
    confidence = float(probabilities[best_index])

    return {
        "category": category,
        "confidence": round(confidence, 4),
        "needs_review": confidence < 0.55,
    }


if __name__ == "__main__":
    trained_model = train_and_evaluate()
    examples = [
        "I was charged twice for my purchase",
        "My refund has not arrived",
        "The application crashes whenever I open it",
        "I forgot my password and cannot log in",
        "My parcel has not been delivered",
    ]
    print("\\nExample predictions:")
    for example in examples:
        print(f"\\nTicket: {example}")
        print(predict_category(example, trained_model))
