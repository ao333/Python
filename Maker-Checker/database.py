from sqlalchemy import create_engine, Column, Integer, String, Enum, ForeignKey, Text, Float, Boolean
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
import enum

# Create an engine that stores data in the local directory's sqlite database file.
engine = create_engine('sqlite:///transactions.db', echo=False)
Base = declarative_base()

class Role(enum.Enum):
    MAKER = "Maker"
    CHECKER = "Checker"
    ADMIN = "Admin"

class User(Base):
    """Model for users table."""
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    role = Column(Enum(Role), nullable=False)

    holdings = relationship("Holding", back_populates="user")
    approvals = relationship("TransactionApproval", back_populates="checker")

    def __repr__(self):
        return f"<User(username='{self.username}', role='{self.role}')>"

class FinancialInstrument(Base):
    """Model for financial instruments table."""
    __tablename__ = 'financial_instruments'

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)

    holdings = relationship("Holding", back_populates="instrument")

    def __repr__(self):
        return f"<FinancialInstrument(name='{self.name}')>"

class Holding(Base):
    """Model for holdings table."""
    __tablename__ = 'holdings'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    instrument_id = Column(Integer, ForeignKey('financial_instruments.id'))
    quantity = Column(Float, default=0)

    user = relationship("User", back_populates="holdings")
    instrument = relationship("FinancialInstrument", back_populates="holdings")

    def __repr__(self):
        return f"<Holding(user_id='{self.user_id}', instrument_id='{self.instrument_id}', quantity='{self.quantity}')>"

class TransactionType(enum.Enum):
    SUBSCRIPTION = "Subscription"
    REDEMPTION = "Redemption"
    SWITCHING = "Switching"

class TransactionStatus(enum.Enum):
    PENDING = "Pending"
    APPROVED = "Approved"
    REJECTED = "Rejected"
    CANCELED = "Canceled"
    UNDER_REVIEW = "Under Review"

class Transaction(Base):
    """Model for transactions table."""
    __tablename__ = 'transactions'

    id = Column(Integer, primary_key=True)
    transaction_type = Column(Enum(TransactionType), nullable=False)
    status = Column(Enum(TransactionStatus), default=TransactionStatus.PENDING)
    maker_id = Column(Integer, ForeignKey('users.id'))
    from_instrument_id = Column(Integer, ForeignKey('financial_instruments.id'), nullable=True)
    to_instrument_id = Column(Integer, ForeignKey('financial_instruments.id'), nullable=True)
    quantity = Column(Float, default=0)
    details = Column(Text)
    maker = relationship("User")
    notes = relationship("Note", back_populates="transaction")
    from_instrument = relationship("FinancialInstrument", foreign_keys=[from_instrument_id])
    to_instrument = relationship("FinancialInstrument", foreign_keys=[to_instrument_id])
    approvals = relationship("TransactionApproval", back_populates="transaction")

    def __repr__(self):
        return f"<Transaction(id='{self.id}', type='{self.transaction_type}', status='{self.status}')>"

class TransactionApproval(Base):
    """Many-many relations for multiple checkers"""
    __tablename__ = 'transaction_approvals'

    id = Column(Integer, primary_key=True)
    transaction_id = Column(Integer, ForeignKey('transactions.id'))
    checker_id = Column(Integer, ForeignKey('users.id'))
    approved = Column(Boolean, nullable=False)
    comments = Column(Text, nullable=True)
    transaction = relationship("Transaction", back_populates="approvals")
    checker = relationship("User", back_populates="approvals")

    def __repr__(self):
        status = 'Approved' if self.approved else 'Rejected'
        return f"<TransactionApproval(transaction_id='{self.transaction_id}', checker_id='{self.checker_id}', status='{status}')>"

class Note(Base):
    """Model for notes table."""
    __tablename__ = 'notes'

    id = Column(Integer, primary_key=True)
    transaction_id = Column(Integer, ForeignKey('transactions.id'))
    user_id = Column(Integer, ForeignKey('users.id'))
    content = Column(Text)
    transaction = relationship("Transaction", back_populates="notes")
    user = relationship("User")

    def __repr__(self):
        return f"<Note(id='{self.id}', user_id='{self.user_id}', transaction_id='{self.transaction_id}')>"

class AuditTrail(Base):
    """Model for audit trail table."""
    __tablename__ = 'audit_trail'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    action = Column(String)
    details = Column(Text)
    user = relationship("User")

    def __repr__(self):
        return f"<AuditTrail(user_id='{self.user_id}', action='{self.action}')>"

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)