from database import Session, User, Role, Transaction, TransactionType, TransactionStatus, Note, AuditTrail, FinancialInstrument, Holding, TransactionApproval
from sqlalchemy.exc import NoResultFound

class Application:
    """Main application class handling business logic."""
    def __init__(self):
        self.session = Session()
        # Define the number of required approvals/checkers.
        self.required_approvals = 2

    def create_user(self, username, role):
        """Admin can create a new user."""
        print(f"Creating user '{username}' with role '{role}'")
        # Check if user already exists.
        if self.session.query(User).filter_by(username=username).first():
            print("User already exists.")
            return
        # Create and add new user.
        user = User(username=username, role=Role(role))
        self.session.add(user)
        self.session.commit()
        self.log_audit(user.id, "Create User", f"Created user with role {role}")

    def edit_user_role(self, username, new_role):
        """Admin can edit permissions of existing users."""
        print(f"Editing role of user '{username}' to '{new_role}'")
        user = self.session.query(User).filter_by(username=username).first()
        if user:
            user.role = Role(new_role)
            self.session.commit()
            self.log_audit(user.id, "Edit User Role", f"Changed role to {new_role}")
        else:
            print("User not found.")

    def remove_user(self, username):
        """Admin can remove a user from the system."""
        print(f"Removing user '{username}'")
        user = self.session.query(User).filter_by(username=username).first()
        if user:
            self.session.delete(user)
            self.session.commit()
            self.log_audit(user.id, "Remove User", "User removed from system")
        else:
            print("User not found.")

    def log_audit(self, user_id, action, details):
        """Log actions in the audit trail."""
        audit = AuditTrail(user_id=user_id, action=action, details=details)
        self.session.add(audit)
        self.session.commit()

    def maker_create_transaction(self, maker_username, transaction_type, instrument_name, quantity, details):
        """Maker initializes an approval workflow for a transaction."""
        print(f"Maker '{maker_username}' is creating a '{transaction_type}' transaction for '{instrument_name}'")
        maker = self.get_user(maker_username)
        if maker and maker.role == Role.MAKER:
            instrument = self.get_or_create_instrument(instrument_name)
            if transaction_type == "Subscription":
                transaction = Transaction(
                    transaction_type=TransactionType(transaction_type),
                    maker_id=maker.id,
                    to_instrument_id=instrument.id,
                    quantity=quantity,
                    details=details
                )
            elif transaction_type == "Redemption":
                transaction = Transaction(
                    transaction_type=TransactionType(transaction_type),
                    maker_id=maker.id,
                    from_instrument_id=instrument.id,
                    quantity=quantity,
                    details=details
                )
            else:
                # For Switching, we need both from_instrument and to_instrument
                print("Switching transactions require both from and to instruments.")
                return
            self.session.add(transaction)
            self.session.commit()
            self.log_audit(maker.id, "Create Transaction", f"Transaction ID {transaction.id} created")
            print(f"Transaction ID {transaction.id} created")
        else:
            print("Invalid maker or user does not have Maker role.")

    def maker_create_switching_transaction(self, maker_username, from_instrument_name, to_instrument_name, quantity, details):
        """Maker initializes a switching transaction."""
        print(f"Maker '{maker_username}' is creating a 'Switching' transaction from '{from_instrument_name}' to '{to_instrument_name}'")
        maker = self.get_user(maker_username)
        if maker and maker.role == Role.MAKER:
            from_instrument = self.get_or_create_instrument(from_instrument_name)
            to_instrument = self.get_or_create_instrument(to_instrument_name)
            transaction = Transaction(
                transaction_type=TransactionType.SWITCHING,
                maker_id=maker.id,
                from_instrument_id=from_instrument.id,
                to_instrument_id=to_instrument.id,
                quantity=quantity,
                details=details
            )
            self.session.add(transaction)
            self.session.commit()
            self.log_audit(maker.id, "Create Switching Transaction", f"Transaction ID {transaction.id} created")
            print(f"Transaction ID {transaction.id} created")
        else:
            print("Invalid maker or user does not have Maker role.")

    def maker_edit_transaction(self, maker_username, transaction_id, new_details):
        """Maker edits a pending transaction."""
        print(f"Maker '{maker_username}' is editing transaction ID '{transaction_id}'")
        maker = self.get_user(maker_username)
        transaction = self.get_transaction(transaction_id)
        if maker and maker.role == Role.MAKER and transaction.maker_id == maker.id and transaction.status == TransactionStatus.PENDING:
            transaction.details = new_details
            self.session.commit()
            self.log_audit(maker.id, "Edit Transaction", f"Transaction ID {transaction.id} edited")
            print(f"Transaction ID {transaction.id} edited")
        else:
            print("Invalid maker, transaction not found, or transaction not editable.")

    def maker_cancel_transaction(self, maker_username, transaction_id):
        """Maker cancels a pending transaction."""
        print(f"Maker '{maker_username}' is canceling transaction ID '{transaction_id}'")
        maker = self.get_user(maker_username)
        transaction = self.get_transaction(transaction_id)
        if maker and transaction.maker_id == maker.id and transaction.status == TransactionStatus.PENDING:
            transaction.status = TransactionStatus.CANCELED
            self.session.commit()
            self.log_audit(maker.id, "Cancel Transaction", f"Transaction ID {transaction.id} canceled")
            print(f"Transaction ID {transaction.id} canceled")
        else:
            print("Invalid maker, transaction not found, or transaction not pending.")

    def checker_approve_transaction(self, checker_username, transaction_id, comments=None):
        """Checker approves a pending transaction. There can be more than 1 checker"""
        print(f"Checker '{checker_username}' is approving transaction ID '{transaction_id}'")
        checker = self.get_user(checker_username)
        transaction = self.get_transaction(transaction_id)
        if checker and checker.role == Role.CHECKER and transaction.status in [TransactionStatus.PENDING, TransactionStatus.UNDER_REVIEW]:
            # Check if checker has already approved/rejected
            existing_approval = self.session.query(TransactionApproval).filter_by(transaction_id=transaction.id, checker_id=checker.id).first()
            if existing_approval:
                print("Checker has already reviewed this transaction.")
                return
            # Create approval record
            approval = TransactionApproval(transaction_id=transaction.id, checker_id=checker.id, approved=True, comments=comments)
            self.session.add(approval)
            self.session.commit()
            self.log_audit(checker.id, "Approve Transaction", f"Transaction ID {transaction.id} approved by checker {checker.username}")
            print(f"Transaction ID {transaction.id} approved by checker {checker.username}")
            # Update transaction status if necessary
            self.update_transaction_status(transaction)
        else:
            print("Invalid checker, transaction not found, or transaction not pending.")

    def checker_reject_transaction(self, checker_username, transaction_id, comments=None):
        """Checker rejects a pending transaction. There can be more than 1 checker"""
        print(f"Checker '{checker_username}' is rejecting transaction ID '{transaction_id}'")
        checker = self.get_user(checker_username)
        transaction = self.get_transaction(transaction_id)
        if checker and checker.role == Role.CHECKER and transaction.status in [TransactionStatus.PENDING, TransactionStatus.UNDER_REVIEW]:
            # Check if checker has already approved/rejected
            existing_approval = self.session.query(TransactionApproval).filter_by(transaction_id=transaction.id, checker_id=checker.id).first()
            if existing_approval:
                print("Checker has already reviewed this transaction.")
                return
            # Create rejection record
            approval = TransactionApproval(transaction_id=transaction.id, checker_id=checker.id, approved=False, comments=comments)
            self.session.add(approval)
            self.session.commit()
            self.log_audit(checker.id, "Reject Transaction", f"Transaction ID {transaction.id} rejected by checker {checker.username}")
            print(f"Transaction ID {transaction.id} rejected by checker {checker.username}")
            # Update transaction status if necessary
            transaction.status = TransactionStatus.REJECTED
            self.session.commit()
            print(f"Transaction ID {transaction.id} has been rejected.")
        else:
            print("Invalid checker, transaction not found, or transaction not pending.")

    def update_transaction_status(self, transaction):
        """Update the transaction status based on approvals."""
        approvals = self.session.query(TransactionApproval).filter_by(transaction_id=transaction.id, approved=True).count()
        print(f"Transaction ID {transaction.id} has {approvals} approvals.")
        if approvals >= self.required_approvals:
            # Update holdings based on transaction type
            self.update_holdings(transaction)
            transaction.status = TransactionStatus.APPROVED
            self.session.commit()
            print(f"Transaction ID {transaction.id} has been approved.")
        else:
            transaction.status = TransactionStatus.UNDER_REVIEW
            self.session.commit()
            print(f"Transaction ID {transaction.id} is under review.")

    def add_note_to_transaction(self, username, transaction_id, content):
        """Adds a note to a transaction."""
        print(f"User '{username}' is adding a note to transaction ID '{transaction_id}'")
        user = self.get_user(username)
        transaction = self.get_transaction(transaction_id)
        if user and transaction:
            note = Note(transaction_id=transaction.id, user_id=user.id, content=content)
            self.session.add(note)
            self.session.commit()
            self.log_audit(user.id, "Add Note", f"Note added to transaction ID {transaction.id}")
            print(f"Note added to transaction ID {transaction.id}")
        else:
            print("User or transaction not found.")

    def update_holdings(self, transaction):
        """Update user holdings based on the transaction."""
        maker = transaction.maker
        if transaction.transaction_type == TransactionType.SUBSCRIPTION:
            # Add quantity to maker's holdings for the instrument
            holding = self.get_or_create_holding(maker.id, transaction.to_instrument_id)
            holding.quantity += transaction.quantity
        elif transaction.transaction_type == TransactionType.REDEMPTION:
            # Subtract quantity from maker's holdings for the instrument
            holding = self.get_or_create_holding(maker.id, transaction.from_instrument_id)
            if holding.quantity >= transaction.quantity:
                holding.quantity -= transaction.quantity
            else:
                print("Insufficient holdings for redemption.")
                return
        elif transaction.transaction_type == TransactionType.SWITCHING:
            # Subtract quantity from from_instrument and add to to_instrument
            from_holding = self.get_or_create_holding(maker.id, transaction.from_instrument_id)
            to_holding = self.get_or_create_holding(maker.id, transaction.to_instrument_id)
            if from_holding.quantity >= transaction.quantity:
                from_holding.quantity -= transaction.quantity
                to_holding.quantity += transaction.quantity
            else:
                print("Insufficient holdings for switching.")
                return
        else:
            print("Unknown transaction type.")
            return
        self.session.commit()

    def get_or_create_holding(self, user_id, instrument_id):
        """Retrieve or create a holding."""
        holding = self.session.query(Holding).filter_by(user_id=user_id, instrument_id=instrument_id).first()
        if not holding:
            holding = Holding(user_id=user_id, instrument_id=instrument_id, quantity=0)
            self.session.add(holding)
            self.session.commit()
        return holding

    def get_or_create_instrument(self, instrument_name):
        """Retrieve or create a financial instrument."""
        instrument = self.session.query(FinancialInstrument).filter_by(name=instrument_name).first()
        if not instrument:
            instrument = FinancialInstrument(name=instrument_name)
            self.session.add(instrument)
            self.session.commit()
        return instrument

    def get_user(self, username):
        """Retrieve a user by username."""
        try:
            return self.session.query(User).filter_by(username=username).one()
        except NoResultFound:
            print(f"User '{username}' not found.")
            return None

    def get_transaction(self, transaction_id):
        """Retrieve a transaction by ID."""
        return self.session.query(Transaction).filter_by(id=transaction_id).first()

    def view_transactions(self, username, status_filter=None):
        """Users can view transactions based on their role and filter."""
        user = self.get_user(username)
        if not user:
            return []
        print(f"User '{username}' is viewing transactions")
        query = self.session.query(Transaction)
        if user.role == Role.MAKER:
            query = query.filter_by(maker_id=user.id)
        elif user.role == Role.CHECKER:
            # For checkers, show transactions that are pending or under review and that they haven't already reviewed
            reviewed_transaction_ids = [approval.transaction_id for approval in user.approvals]
            query = query.filter(Transaction.id.notin_(reviewed_transaction_ids))
            query = query.filter(Transaction.status.in_([TransactionStatus.PENDING, TransactionStatus.UNDER_REVIEW]))
        if status_filter:
            query = query.filter_by(status=TransactionStatus(status_filter))
        transactions = query.all()
        for t in transactions:
            print(t)
        return transactions

    def view_holdings(self, username):
        """Display holdings for a user."""
        user = self.get_user(username)
        if not user:
            return
        print(f"Holdings for user '{username}':")
        holdings = self.session.query(Holding).filter_by(user_id=user.id).all()
        for holding in holdings:
            instrument_name = holding.instrument.name
            print(f" - {instrument_name}: {holding.quantity} shares")