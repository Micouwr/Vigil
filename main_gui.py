# main_gui.py
import tkinter as tk
from tkinter import scrolledtext, messagebox
from bot_engine.router import classify_ticket, _validate_metadata
import threading


def build_status_text(result):
    """Composes the status line from a classification result.

    The status combines a processing part and a persistence part as
    separate pieces of information. The processing part is determined
    first, from the result, in priority order: PII preprocessing
    failure, then API failure, then meaningful-text skip, then the
    successful classification wording. Each processing message
    activates only when the corresponding flag is the actual Boolean
    True; a present value that is not the actual Boolean True does not
    activate the message. A routing claim is made only when the
    routing decision confirms the human_agent route; a routing
    decision that is not a dictionary has no route to confirm and is
    treated as unconfirmed. The persistence part is appended second,
    from record_status. A persistence failure never hides or
    contradicts a processing failure. "Persistence not confirmed"
    does not mean the record is absent: a partial write may have occurred.
    """
    routing = result.get("routing_decision")
    route = routing.get("route") if isinstance(routing, dict) else None
    human_confirmed = route == "human_agent"

    if result.get("preprocessing_failed") is True:
        if human_confirmed:
            processing = "PII preprocessing failed; routed to human review."
        else:
            processing = "PII preprocessing failed."
    elif result.get("api_failed") is True:
        if human_confirmed:
            processing = "API call failed; ticket routed to human review."
        else:
            processing = "API call failed."
    elif result.get("model_skip") is True:
        if human_confirmed:
            processing = "No meaningful text to classify; routed to human review."
        else:
            processing = "No meaningful text to classify."
    else:
        processing = "Classification complete."

    record_status = result.get("record_status")
    if record_status == "record_failed":
        persistence = " Persistence not confirmed."
    elif record_status == "recorded":
        persistence = " Recorded."
    else:
        persistence = " Persistence unknown."

    return processing + persistence


def _pii_display(result):
    """Yes for Boolean True, No for Boolean False, Unknown for missing or malformed.

    Unknown means the value does not establish whether PII is present
    or absent; it is not a No.
    """
    value = result.get("contains_pii")
    if value is True:
        return "Yes"
    if value is False:
        return "No"
    return "Unknown"


class TicketClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vigil")
        self.root.geometry("600x560")

        # --- UI Elements ---
        self.main_frame = tk.Frame(root, padx=10, pady=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Input Area
        input_label = tk.Label(self.main_frame, text="Enter Ticket Text:", font=("Arial", 12))
        input_label.pack(anchor="w")

        self.input_text = scrolledtext.ScrolledText(self.main_frame, height=10, width=70, wrap=tk.WORD)
        self.input_text.pack(pady=5, fill=tk.BOTH, expand=True)

        # Metadata frame: optional fields for demo and testing only.
        # In the future automatic flow, subject and sender come from
        # email headers, not from typing. There is no input field for
        # the Vigil decision ID: Vigil assigns it at intake.
        metadata_frame = tk.LabelFrame(self.main_frame, text="Metadata (optional)", padx=10, pady=10)
        metadata_frame.pack(fill=tk.X, expand=True, pady=5)

        source_label = tk.Label(metadata_frame, text="Source Ticket ID (optional):", font=("Arial", 11))
        source_label.pack(anchor="w")
        self.source_ticket_id_entry = tk.Entry(metadata_frame, font=("Arial", 11))
        self.source_ticket_id_entry.pack(fill=tk.X, pady=2)

        subject_label = tk.Label(metadata_frame, text="Subject (optional):", font=("Arial", 11))
        subject_label.pack(anchor="w")
        self.subject_entry = tk.Entry(metadata_frame, font=("Arial", 11))
        self.subject_entry.pack(fill=tk.X, pady=2)

        sender_label = tk.Label(metadata_frame, text="Sender (optional):", font=("Arial", 11))
        sender_label.pack(anchor="w")
        self.sender_entry = tk.Entry(metadata_frame, font=("Arial", 11))
        self.sender_entry.pack(fill=tk.X, pady=2)

        # Button
        self.classify_button = tk.Button(
            self.main_frame,
            text="Classify Ticket",
            command=self.start_classification_thread,
            font=("Arial", 12, "bold")
        )
        self.classify_button.pack(pady=10)

        # Results Area
        results_frame = tk.LabelFrame(self.main_frame, text="Classification Results", padx=10, pady=10)
        results_frame.pack(fill=tk.X, expand=True)

        self.decision_id_text = tk.StringVar(value="Decision ID: N/A")
        decision_id_label = tk.Label(results_frame, textvariable=self.decision_id_text, font=("Arial", 11))
        decision_id_label.pack(anchor="w")

        self.result_text = tk.StringVar(value="Result: N/A")
        result_label = tk.Label(results_frame, textvariable=self.result_text, font=("Arial", 11))
        result_label.pack(anchor="w")

        self.confidence_text = tk.StringVar(value="Confidence: N/A")
        confidence_label = tk.Label(results_frame, textvariable=self.confidence_text, font=("Arial", 11))
        confidence_label.pack(anchor="w")

        self.pii_text = tk.StringVar(value="PII detected or flagged: N/A")
        pii_label = tk.Label(results_frame, textvariable=self.pii_text, font=("Arial", 11))
        pii_label.pack(anchor="w")

        self.routing_text = tk.StringVar(value="Routing: N/A")
        routing_label = tk.Label(results_frame, textvariable=self.routing_text, font=("Arial", 11))
        routing_label.pack(anchor="w")

        # Status Bar
        self.status_text = tk.StringVar(value="Ready")
        status_bar = tk.Label(root, textvariable=self.status_text, bd=1, relief=tk.SUNKEN, anchor="w")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def start_classification_thread(self):
        """Starts the classification in a separate thread to keep the GUI responsive.

        Captures the ticket text and the metadata together at the
        Classify click, validates the metadata before starting the
        worker thread, and resets the Decision ID label to N/A after
        successful pre-flight validation, so a previous attempt's ID
        cannot remain visible if the new attempt raises before
        returning a result. The worker thread receives the captured
        snapshot rather than reading live GUI fields later.
        """
        ticket_text = self.input_text.get("1.0", tk.END).strip()

        if not ticket_text:
            messagebox.showwarning("Input Error", "Ticket text cannot be empty.")
            return

        try:
            subject, sender, source_ticket_id = _validate_metadata(
                self.subject_entry.get(),
                self.sender_entry.get(),
                self.source_ticket_id_entry.get(),
            )
        except (TypeError, ValueError) as e:
            message = str(e)
            for field_name, label in (
                ("source_ticket_id", "Source Ticket ID"),
                ("subject", "Subject"),
                ("sender", "Sender"),
            ):
                message = message.replace(field_name, label)
            messagebox.showwarning("Input Error", message)
            return

        self.decision_id_text.set("Decision ID: N/A")
        self.classify_button.config(state=tk.DISABLED)
        self.status_text.set("Classifying, please wait...")

        # Run classification in a thread to prevent freezing the GUI
        thread = threading.Thread(
            target=self.run_classification,
            args=(ticket_text, subject, sender, source_ticket_id),
            daemon=True,
        )
        thread.start()

    def run_classification(self, ticket_text, subject=None, sender=None, source_ticket_id=None):
        """Calls the backend classifier and updates the GUI with the results."""
        try:
            result = classify_ticket(
                ticket_text,
                subject=subject,
                sender=sender,
                source_ticket_id=source_ticket_id,
            )
            self.root.after(0, self.update_ui_with_results, result)
        except Exception as e:
            self.root.after(0, self.show_error, str(e))
        finally:
            self.root.after(0, self.reset_ui_state)

    def update_ui_with_results(self, result):
        """Updates the result labels in the GUI."""
        # Decision ID: N/A when missing, None, or empty, never None or empty.
        decision_id = result.get("decision_id") or "N/A"
        self.decision_id_text.set(f"Decision ID: {decision_id}")

        # Safely handle ticket_type
        ticket_type = result.get("ticket_type", "Error")
        self.result_text.set(f"Result: {ticket_type}")

        # Safely handle confidence_score
        conf_score = result.get("confidence_score", 0.0)
        if isinstance(conf_score, (int, float)):
            self.confidence_text.set(f"Confidence: {conf_score * 100:.2f}%")
        else:
            self.confidence_text.set("Confidence: Error")

        # PII display uses strict Boolean identity, with Unknown for
        # missing or malformed values. Yes means PII was detected or
        # conservatively treated as present after a preprocessing
        # failure; it does not by itself establish confirmed detection.
        self.pii_text.set(f"PII detected or flagged: {_pii_display(result)}")

        # Safely handle routing decision
        routing = result.get("routing_decision", {})
        if not isinstance(routing, dict):
            routing = {}
        route = routing.get("route", "unknown")
        reason = routing.get("reason", "")
        if reason:
            self.routing_text.set(f"Routing: {route} ({reason})")
        else:
            self.routing_text.set(f"Routing: {route}")

        self.status_text.set(build_status_text(result))

    def show_error(self, error_message):
        """Shows an error message in the status bar and a popup."""
        self.status_text.set(f"Error: {error_message}")
        messagebox.showerror("Error", f"An unexpected error occurred: {error_message}")

    def reset_ui_state(self):
        """Resets the button to an active state."""
        self.classify_button.config(state=tk.NORMAL)


if __name__ == "__main__":
    root = tk.Tk()
    app = TicketClassifierApp(root)
    root.mainloop()
