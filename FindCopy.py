import psycopg2
import csv

conn = psycopg2.connect(
    dbname="cmsdb",  
    user="cmsuser",            
    password="cmsuser",     
    host="localhost",           
    port="5432"                   
)

cur = conn.cursor()

# 用來儲存查詢結果的列表
results = []

# 用來追蹤每個 user 在各個 contest 下的最高分數
user_scores = {}

# 查詢 submission_results 資料，根據 submission_id 獲得 score 和 submission_id
cur.execute("SELECT submission_id, score FROM submission_results")
submission_results = cur.fetchall()

for submission_result in submission_results:
    submission_id = submission_result[0]  # 提取 submission_id
    score = submission_result[1]  # 提取 score

    # 如果 score 為 0，跳過該次處理
    if score == 0.0:
        continue

    # 根據 submission_id 查詢 submissions 資料，獲取 participation_id, task_id, timestamp
    cur.execute("SELECT participation_id, task_id, timestamp FROM submissions WHERE id = %s", (submission_id,))
    submission_data = cur.fetchone()

    if submission_data:
        participation_id = submission_data[0]  # 提取 participation_id
        task_id = submission_data[1]  # 提取 task_id
        timestamp = submission_data[2]  # 提取 timestamp

        # 根據 participation_id 查詢 participations 資料，獲取 contest_id 和 user_id
        cur.execute("SELECT contest_id, user_id FROM participations WHERE id = %s", (participation_id,))
        participation_data = cur.fetchone()

        if participation_data:
            contest_id = participation_data[0]  # 提取 contest_id
            user_id = participation_data[1]  # 提取 user_id

            # 根據 contest_id 查詢 contests 資料，獲取 contest_name 和 start
            cur.execute("SELECT name, start FROM contests WHERE id = %s", (contest_id,))
            contest_data = cur.fetchone()

            if contest_data:
                contest_name = contest_data[0]  # 提取 contest_name
                contest_start = contest_data[1]  # 提取 contest_start
            else:
                contest_name = "Unknown"
                contest_start = "Unknown"

            # 根據 user_id 查詢 users 資料，獲取 first_name
            cur.execute("SELECT first_name FROM users WHERE id = %s", (user_id,))
            user_data = cur.fetchone()

            if user_data:
                first_name = user_data[0]  # 提取 first_name
            else:
                first_name = "Unknown"

            # 檢查並更新該 user 在各個 contest 的最高分數
            if user_id not in user_scores:
                user_scores[user_id] = {
                    "first_name": first_name,
                    "total_score": score,  # 初始化時設為當前 task 的分數
                    "task_scores": {task_id: score},  # 用來儲存每個 task 的最高分
                    "timestamps": {task_id: timestamp},  # 用來儲存每個 task 的最新 timestamp
                }
            else:
                # 如果該 user 在該 task_id 已經出現，檢查並更新最高分數
                if task_id not in user_scores[user_id]["task_scores"]:
                    user_scores[user_id]["task_scores"][task_id] = score
                    user_scores[user_id]["timestamps"][task_id] = timestamp
                else:
                    # 只更新最高分數，並保留最新的 timestamp
                    if score > user_scores[user_id]["task_scores"][task_id]:
                        user_scores[user_id]["task_scores"][task_id] = score
                        user_scores[user_id]["timestamps"][task_id] = timestamp
                    elif score == user_scores[user_id]["task_scores"][task_id]:
                        # 如果分數相同，保留較新的 timestamp
                        if timestamp > user_scores[user_id]["timestamps"][task_id]:
                            user_scores[user_id]["timestamps"][task_id] = timestamp

                # 更新該 user 的總分，將該 user 在不同 contest 中的 task_id 最高分加總
                user_scores[user_id]["total_score"] = sum(user_scores[user_id]["task_scores"].values())

# 提取每個 user 的結果
final_results = []

for user_id, data in user_scores.items():
    total_score = data["total_score"]
    first_name = data["first_name"]
    
    # 按照 task_scores 取每個 task_id 的最新 timestamp
    latest_timestamp = max(data["timestamps"].values()).strftime("%Y-%m-%d %H:%M:%S")  # 格式化 timestamp 顯示
    
    final_results.append({
        "first_name": first_name,
        "total_score": total_score,
        "timestamp": latest_timestamp
    })

# 按照 first_name 排序結果
final_results_sorted = sorted(final_results, key=lambda x: x['first_name'])

# 寫入 CSV 檔案
with open("output.csv", "w", newline="") as csvfile:
    # 設定最終要寫入 CSV 的欄位名稱
    fieldnames = ["first_name", "total_score", "timestamp"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # 寫入 CSV 標題
    writer.writeheader()

    # 寫入每條資料（包括 timestamp）
    for result in final_results_sorted:
        writer.writerow(result)

# 關閉游標和資料庫連接
cur.close()
conn.close()

print("CSV檔案已經生成：output.csv")
