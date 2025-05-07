import csv
import uuid
import random
from datetime import datetime, timedelta

NUM_TXNS = 150

# CSV file paths
PERSONAL_INFO_CSV = './data/personal_info.csv'
TRANSACTIONS_CSV = './data/transactions.csv'
LOAN_HISTORY_CSV = './data/loan_history.csv'

def random_date(start: datetime, end: datetime) -> str:
    """Return a random date (ISO format) between start and end."""
    delta = end - start
    random_days = random.randint(0, delta.days)
    return (start + timedelta(days=random_days)).date().isoformat()

def generate_transactions(count: int, account_number: str, start: datetime, end: datetime) -> list[dict]:
    """Generate a list of transaction dicts matching the schema."""
    descriptions = [
        'Grocery', 'Salary', 'Electricity bill', 'Water bill',
        'Online purchase', 'Restaurant', 'Flight ticket', 'Fuel',
        'Rent', 'Insurance', 'Subscription'
    ]
    txns = []
    for _ in range(count):
        txn_date = random_date(start, end)
        amount = round(random.uniform(-5000, 5000), 2)
        desc = random.choice(descriptions)
        txn_type = 'credit' if amount > 0 else 'debit'
        txns.append({
            'account_number': account_number,
            'date': txn_date,
            'description': desc,
            'amount': abs(amount),
            'type': txn_type
        })
    return txns

def write_csv(filename, fieldnames, rows):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def populate_csv():
    # 1) single user with random values
    nric = 'S1234567A'
    name = 'Alice Johnson'
    age = random.randint(25, 60)
    income = random.randint(3000, 10000)
    credit_score = random.randint(600, 850)
    account_number = 'ACCT' + str(random.randint(100000, 999999))
    bank_balance = round(random.uniform(1000, 50000), 2)
    mobile_number = '9' + str(random.randint(1000000, 9999999))
    email = 'alice.johnson@example.com'
    personal_info = [{
        'nric': nric,
        'name': name,
        'age': age,
        'income': income,
        'credit_score': credit_score,
        'account_number': account_number,
        'bank_balance': bank_balance,
        'mobile_number': mobile_number,
        'email': email
    }]
    write_csv(PERSONAL_INFO_CSV, [
        'nric', 'name', 'age', 'income', 'credit_score', 'account_number', 'bank_balance', 'mobile_number', 'email'
    ], personal_info)

    # 2) generate and insert random transactions
    start_dt = datetime(2024, 1, 1)
    end_dt   = datetime(2025, 5, 7)
    txns = generate_transactions(NUM_TXNS, account_number, start_dt, end_dt)
    write_csv(TRANSACTIONS_CSV, ['account_number', 'date', 'description', 'amount', 'type'], txns)

    # 3) static loan history
    loan_types = ['Home Loan', 'Car Loan', 'Personal Loan']
    loans = [
        {
            'account_number': account_number,
            'loan_type': random.choice(loan_types),
            'loan_amount': 250000,
            'status': 'Active',
            'monthly_emi': 1200,
            'start_date': '2021-06-01'
        },
        {
            'account_number': account_number,
            'loan_type': random.choice(loan_types),
            'loan_amount': 50000,
            'status': 'Closed',
            'monthly_emi': 850,
            'start_date': '2019-01-01'
        }
    ]
    write_csv(LOAN_HISTORY_CSV, ['account_number', 'loan_type', 'loan_amount', 'status', 'monthly_emi', 'start_date'], loans)


def main():
    print("Populating CSV files with sample data...")
    populate_csv()
    print(f"✔ Wrote personal_info.csv, transactions.csv, loan_history.csv.")
    print("Done.")

if __name__ == '__main__':
    main()
