
from datetime import datetime
from io import StringIO
import pandas as pd
from sqlalchemy.exc import IntegrityError, NoSuchTableError
from flask import jsonify
from models import DataTraining, ModelVersi, db
from io import StringIO
import pandas as pd
from sqlalchemy import Column, VARCHAR, Table, inspect, text

def get_latest_version_table_info():
    # get all table 
    inspector = inspect(db.engine)
    all_tables_1 = inspector.get_table_names()

    print("All Tables in Metadata:", all_tables_1)
    all_tables = inspect(db.engine).get_table_names()

    # get table with relevant name
    relevant_tables = [table for table in all_tables if table.startswith("dt_v")]

    # get latest table version
    if relevant_tables:
        latest_version_table_name = max(relevant_tables)
        return get_table_info(latest_version_table_name)

    return None

def get_table_info(table_name):

    if table_name:
        # use inspector for get columns
        inspector = inspect(db.engine)
        columns = inspector.get_columns(table_name)
        
        # only columns
        column_names = [column["name"] for column in columns]

        return {"table_name": table_name, "columns": column_names}

    return None

def create_dynamic_columns(df, table_name):
    try:
        columns = df.columns

        # create new table
        dynamic_columns = [Column(col.strip(), VARCHAR(255)) for col in columns]
        dynamic_table = Table(table_name, db.metadata, *dynamic_columns, extend_existing=True)

        # Create the table in the database
        dynamic_table.create(bind=db.engine, checkfirst=True)

    except Exception as e:
        raise e

def merge_data_training(df, table_name):
    try:
        latest_version_table_info = get_latest_version_table_info()

        print(latest_version_table_info['table_name'])
        print(latest_version_table_info['columns'])
        if not latest_version_table_info:
            return jsonify({"error": "Merging data training failed, data training is empty. You should perform an update first."}), 400

        # Check if the columns match
        if set(df.columns) != set(latest_version_table_info['columns']):
            return jsonify({"error": "Merging data training failed, data training is not synchronized. You should perform an update."}), 400

        # Create a new table with dynamic columns
        create_dynamic_columns(df, table_name)

        # Merge the data and insert into the new table
        merged_df = pd.concat([pd.read_sql_table(latest_version_table_info['table_name'], db.engine), df], ignore_index=True)
        merged_df.to_sql(table_name, db.engine, if_exists='replace', index=False)

        return jsonify({"message": "Data training merged successfully"}), 200

    except IntegrityError as e:
        db.session.rollback()
        return jsonify({"error": "Merging data training failed. Integrity error."}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 502

def update_data_training(df, table_name):
    try:
        # create table
        create_dynamic_columns(df, table_name)

        # Assign struktur tabel ke objek DataFrame
        df.to_sql(table_name, db.engine, if_exists='replace', index=False)

        return jsonify({"message": "Data training updated successfully"}), 200

    except NoSuchTableError:
        return jsonify({"error": f"Table {table_name} does not exist"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 503
    
def fetch_selected_data(version):
    if version:
        # Fetch data based on the selected version and all previous versions
        selected_data = DataTraining.query.filter(DataTraining.versi <= version).all()
    else:
        # If version is not provided, fetch all data
        selected_data = DataTraining.query.all()

    return selected_data
    
def save_model_version(model_path):
    # Simpan versi model ke database
    new_model_version = ModelVersi(versi_model=f"ML_DT_{datetime.now().strftime('%d%m%Y')}", model_path=model_path)
    db.session.add(new_model_version)
    db.session.commit()

    return new_model_version.versi_model
