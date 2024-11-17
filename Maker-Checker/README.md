python3 -m venv myenv
source my env/bin/activate
source myenv/bin/activate
pip3 install sqlalchemy
run test.py

database.py contains the data schemas.
- Uses Enum for centralized definition of roles and states.
- Uses Object Relational Mapping to produce parameterized queries. This helps prevent SQL Injections by disallowing malicious sql input from being inserted into final db query.
- Creates the engine to create tables and store locally for ease of use/run/debug 

application.py contains the core logic. It allows
- Admins to reate/edit/remove makers/admin/checkers. 
- Makers to create/edit/cancel subscription/redemption/switch transactions
- MULTIPLE checkers to approve/reject transactions
- Anyone to add notes to transactions
- Create instruments and update holdings
- All users to view transactions/holdings
- Audit Logger to log all activities

test.py allows you to test all functions, and print the output to compare against expected results
- It will create a transactions.db file to locally persist data, emulating a SQL database


[User] 1-holds----------* [Holdings] *-Held in-1 [Instrument]
|
|1                            2
|
[TransactionApproval]*-1 [Transaction] 1-* [Note]
*
[User/Checker]