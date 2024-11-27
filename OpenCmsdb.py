import psycopg2
from psycopg2 import sql
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from pandastable import Table

def fetch_table_data(table_name):
    db_config = {
        "dbname": "cmsdb",
        "user": "cmsuser",
        "password": "cmsuser",
        "host": "localhost",
        "port": "5432"
    }

    try:
        connection = psycopg2.connect(**db_config)
        cursor = connection.cursor()
        query = sql.SQL("SELECT * FROM {table_name};").format(table_name=sql.Identifier(table_name))
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(rows, columns=columns)
        display_csv_table(df, table_name)

    except psycopg2.Error as e:
        messagebox.showerror("資料庫錯誤", f"無法讀取資料表 {table_name}：\n{e}")
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

def display_csv_table(df, table_name):
    table_window = tk.Toplevel()
    table_window.title(f"資料表: {table_name}")
    frame = ttk.Frame(table_window)
    frame.pack(fill=tk.BOTH, expand=True)
    pt = Table(frame, dataframe=df, showtoolbar=True, showstatusbar=True)
    pt.show()

def create_gui():
    root = tk.Tk()
    root.title("選擇資料表")
    label = tk.Label(root, text="選擇資料夾：", font=("Arial", 14))
    label.pack(pady=10)
    table_names = ["admins", "announcements", "attachments", "contests", "datasets", "evaluations", "executables", "files", 
                   "fsobjects", "managers", "messages", "participations", "printjobs", "questions", "statements", "submission_results",
                   "submissions", "tasks", "teams", "testcases", "tokens", "user_test_executables", "user_test_files", "user_test_managers", 
                   "user_test_results", "user_tests", "users"]
    selected_table = tk.StringVar(value=table_names[0])
    dropdown = ttk.Combobox(root, textvariable=selected_table, values=table_names, state="readonly")
    dropdown.pack(pady=10)
    button = tk.Button(root, text="確定", font=("Arial", 12), command=lambda: fetch_table_data(selected_table.get()))
    button.pack(pady=10)
    root.mainloop()

if __name__ == "__main__":
    create_gui()
