

set -e



DB_NAME="dermai_db"
DB_USER="dermai_user"
DB_PASS="your_password_here"

echo "=== DermAI Classifier — Database Setup ==="
echo "Creating user '$DB_USER' and database '$DB_NAME'..."

psql -U postgres <<-SQL
  DO \$\$
  BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = '$DB_USER') THEN
      CREATE USER $DB_USER WITH PASSWORD '$DB_PASS';
    END IF;
  END
  \$\$;

  SELECT 'CREATE DATABASE $DB_NAME OWNER $DB_USER'
  WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$DB_NAME')\gexec

  GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;
SQL

echo ""
echo " Database setup complete."
echo "   Connection string: postgresql://$DB_USER:$DB_PASS@localhost:5432/$DB_NAME"
echo ""
echo "Next steps:"
echo "  1. Copy .env.example to .env and update DATABASE_URL if needed"
echo "  2. Place your .keras model at: ml/skin_model.keras"
echo "  3. Run: uvicorn api.main:app --reload"
