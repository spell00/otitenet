import argparse
import datetime
import sys
import mysql.connector
from mysql.connector import Error


def connect_db(host: str, user: str, password: str, database: str):
    try:
        conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database,
        )
        return conn
    except Error as e:
        print(f"❌ Connection error: {e}")
        sys.exit(1)


def table_exists(cursor, table_name: str) -> bool:
    cursor.execute("SHOW TABLES LIKE %s", (table_name,))
    return cursor.fetchone() is not None


def truncate_table(cursor, table_name: str):
    cursor.execute(f"TRUNCATE TABLE `{table_name}`")


def backup_and_recreate(cursor, table_name: str, backup_name: str):
    # Rename original to backup
    cursor.execute(f"RENAME TABLE `{table_name}` TO `{backup_name}`")
    # Recreate empty table with the same structure using LIKE
    cursor.execute(f"CREATE TABLE `{table_name}` LIKE `{backup_name}`")


def parse_args():
    parser = argparse.ArgumentParser(description="Manage best_models_registry table: truncate or backup+recreate.")
    parser.add_argument("--action", choices=["truncate", "backup"], required=True,
                        help="Operation to perform: 'truncate' to empty, 'backup' to rename and recreate empty table.")
    parser.add_argument("--table", default="best_models_registry", help="Target table name (default: best_models_registry)")
    parser.add_argument("--backup-name", default=None, help="Backup table name (default: {table}_backup_<timestamp>)")
    parser.add_argument("--host", default="localhost", help="MySQL host (default: localhost)")
    parser.add_argument("--user", default="y_user", help="MySQL user (default: y_user)")
    parser.add_argument("--password", default="password", help="MySQL password (default: password)")
    parser.add_argument("--database", default="results_db", help="MySQL database name (default: results_db)")
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without executing")
    return parser.parse_args()


def main():
    args = parse_args()

    backup_name = args.backup_name
    if args.action == "backup" and not backup_name:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{args.table}_backup_{ts}"

    if args.dry_run:
        if args.action == "truncate":
            print(f"Would TRUNCATE TABLE `{args.table}`")
        else:
            print(f"Would RENAME TABLE `{args.table}` TO `{backup_name}` and CREATE TABLE `{args.table}` LIKE `{backup_name}`")
        return

    conn = connect_db(args.host, args.user, args.password, args.database)
    try:
        cursor = conn.cursor()

        if not table_exists(cursor, args.table):
            print(f"❌ Table '{args.table}' does not exist in database '{args.database}'.")
            sys.exit(1)

        if args.action == "truncate":
            truncate_table(cursor, args.table)
            conn.commit()
            print(f"✅ Truncated table: {args.table}")
        else:
            if table_exists(cursor, backup_name):
                print(f"❌ Backup table '{backup_name}' already exists. Choose a different name with --backup-name.")
                sys.exit(1)
            # Perform backup and recreation
            backup_and_recreate(cursor, args.table, backup_name)
            conn.commit()
            print(f"✅ Backed up '{args.table}' to '{backup_name}' and recreated empty '{args.table}'.")
    except Error as e:
        print(f"❌ Error: {e}")
        conn.rollback()
        sys.exit(1)
    finally:
        try:
            cursor.close()
        except Exception:
            pass
        conn.close()


if __name__ == "__main__":
    main()
