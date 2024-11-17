from application import *

def run_tests():
    app = Application()

    # Admin actions
    app.create_user("alice", "Maker")
    app.create_user("bob", "Checker")
    app.create_user("carol", "Checker")
    app.create_user("admin", "Admin")

    # Create 'XYZ' financial instrument
    xyz_instrument = app.get_or_create_instrument("XYZ")
    alice = app.get_user("alice")
    # Assume users start with 100 shares of 'XYZ' company
    alice_holding = Holding(user_id=alice.id, instrument_id=xyz_instrument.id, quantity=100)
    app.session.add(alice_holding)
    app.session.commit()
    # View holdings
    app.view_holdings("alice")

    # Maker actions
    # Alice creates a Redemption transaction to sell 50 shares of XYZ
    app.maker_create_transaction("alice", "Redemption", "XYZ", 50, "Selling 50 shares of XYZ")
    # Alice adds a note to the transaction
    app.add_note_to_transaction("alice", 1, "Please process quickly")

    # Alice edits the transaction to sell 40 shares instead of 50
    app.maker_edit_transaction("alice", 1, "Selling 40 shares of XYZ")
    # Update the quantity as well
    transaction = app.get_transaction(1)
    transaction.quantity = 40
    app.session.commit()
    # View transactions as Maker
    app.view_transactions("alice")

    # Checker actions
    # Bob adds a note to the transaction
    app.add_note_to_transaction("bob", 1, "Reviewed by Bob")
    # Carol adds a note to the transaction
    app.add_note_to_transaction("carol", 1, "Reviewed by Carol")
    # Bob approves the transaction
    app.checker_approve_transaction("bob", 1)
    # Carol approves the transaction (2 checkers)
    app.checker_approve_transaction("carol", 1)
    # View holdings after transaction approval
    app.view_holdings("alice")

    # Alice creates a Subscription transaction to buy 20 shares of XYZ
    app.maker_create_transaction("alice", "Subscription", "XYZ", 20, "Buying 20 shares of XYZ")
    # Bob rejects the transaction (1 checker)
    app.checker_reject_transaction("bob", 2)
    # View holdings after transaction rejection
    app.view_holdings("alice")

    # Alice creates a Switching transaction from XYZ to ABC
    # First, create ABC instrument
    abc_instrument = app.get_or_create_instrument("ABC")
    # Give Alice 50 starting ABC shares
    alice_abc_holding = Holding(user_id=alice.id, instrument_id=abc_instrument.id, quantity=50)
    app.session.add(alice_abc_holding)
    app.session.commit()
    # Alice wants to switch 30 shares from XYZ to ABC. Minnus 30 ABC from Alice
    app.maker_create_switching_transaction("alice", "XYZ", "ABC", 30, "Switching 30 shares from XYZ to ABC")
    # Bob approves the switching transaction
    app.checker_approve_transaction("bob", 3)
    # Carol approves the switching transaction
    app.checker_approve_transaction("carol", 3)
    # View holdings after switching XYZ: 60-30=30 | ABC: 50+30=80
    app.view_holdings("alice")
    # View transactions as Checker
    app.view_transactions("bob")

    # Maker cancels a transaction
    # Alice creates a Subscription transaction but cancels it before approval
    app.maker_create_transaction("alice", "Subscription", "XYZ", 10, "Buying 10 shares of XYZ")
    app.maker_cancel_transaction("alice", 4)
    # Cancelled, so no change
    app.view_holdings("alice")

    # Admin edits user role
    app.edit_user_role("alice", "Checker")

    # View audit trail (for demonstration)
    print("\nAudit Trail:")
    session = Session()
    audit_entries = session.query(AuditTrail).all()
    for entry in audit_entries:
        print(entry)

if __name__ == "__main__":
    run_tests()