#整理每一筆繳交資料->整理成每個使用者有得分的資料，可以看到得到最高分數，最早繳交的那筆->匯出至Excel
import psycopg2
import pandas as pd
from datetime import datetime, timedelta

# 連接資料庫
conn = psycopg2.connect(
    dbname="cmsdb",
    user="postgres",
    password="postgres",
    host="localhost",
    port="5432"
)

cur = conn.cursor()

# 存每一筆繳交資料，包含使用者，考試名稱，開始時間，繳交時間，分數，題目名稱
user_contest_scores = []

# 查詢 submission_results 資料，根據 submission_id 獲得 score
cur.execute("SELECT submission_id, score FROM submission_results")
submission_results = cur.fetchall()

for submission_result in submission_results:
    submission_id = submission_result[0]  # 提取 submission_id
    score = submission_result[1]         # 提取 score

    # 根據 submission_id 查詢 submissions 資料，獲取 participation_id, task_id, timestamp
    cur.execute("SELECT participation_id, task_id, timestamp FROM submissions WHERE id = %s", (submission_id,))
    submission_data = cur.fetchone()
    if submission_data:
        participation_id = submission_data[0]
        task_id = submission_data[1]
        timestamp = submission_data[2]

        cur.execute("SELECT name FROM tasks WHERE id = %s", (task_id,))
        tasks_data = cur.fetchone()
        if tasks_data:
            task_name = tasks_data[0]

        # 根據 participation_id 查詢 participations 資料，獲取 contest_id 和 user_id
        cur.execute("SELECT contest_id, user_id FROM participations WHERE id = %s", (participation_id,))
        participation_data = cur.fetchone()
        if participation_data:
            contest_id = participation_data[0]
            user_id = participation_data[1]

            # 根據 contest_id 查詢 contests 資料，獲取 contest_name 和 start
            cur.execute("SELECT name, start FROM contests WHERE id = %s", (contest_id,))
            contest_data = cur.fetchone()
            if contest_data:
                contest_name = contest_data[0]
                contest_start = contest_data[1]
            else:
                contest_name = "Unknown"
                contest_start = "Unknown"

            # 根據 user_id 查詢 users 資料，獲取 first_name
            cur.execute("SELECT first_name FROM users WHERE id = %s", (user_id,))
            user_data = cur.fetchone()
            if user_data:
                first_name = user_data[0]
            else:
                first_name = "Unknown"

            # 準備記錄每次提交的資訊
            user_contest_scores.append({
                "使用者": first_name,
                "考試名稱": contest_name,
                "開始時間": contest_start.strftime("%Y-%m-%d %H:%M:%S") if contest_start != "Unknown" else contest_start,
                "繳交時間": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "分數": score,
                "題目名稱": task_name,
            })

# 使用 pandas 將結果轉換為 DataFrame
df = pd.DataFrame(user_contest_scores)

# 過濾掉分數為 0 的資料
df_filtered = df[df["分數"] != 0]

# 先排序資料，先依使用者、考試名稱、題目名稱分組，再按照分數降序排列，若分數相同則按照繳交時間升序排列
df_sorted = df_filtered.sort_values(by=["使用者", "考試名稱", "題目名稱", "分數", "繳交時間"], ascending=[True, True, True, False, True])

# 合併相同使用者、考試名稱、題目名稱的資料
df_grouped = df_sorted.groupby(["使用者", "考試名稱", "題目名稱"], as_index=False).first()

# 將資料輸出到 Excel 檔案
df_grouped.to_excel("TestResult.xlsx", index=False)

# 關閉資料庫連接
cur.close()
conn.close()
