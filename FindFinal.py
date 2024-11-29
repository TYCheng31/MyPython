import psycopg2
import csv
from datetime import datetime, timedelta

# 連接資料庫
conn = psycopg2.connect(
    dbname="cmsdb",
    user="cmsuser",
    password="cmsuser",
    host="localhost",
    port="5432"
)

cur = conn.cursor()

# 用來追蹤每個 user 和 contest_id 的最高分數總和
user_contest_scores = {}

# 查詢 submission_results 資料，根據 submission_id 獲得 score 和 submission_id
cur.execute("SELECT submission_id, score FROM submission_results")
submission_results = cur.fetchall()

for submission_result in submission_results:
    submission_id = submission_result[0]
    score = submission_result[1]

    # 根據 submission_id 查詢 submissions 資料，獲取 participation_id, task_id, timestamp
    cur.execute("SELECT participation_id, task_id, timestamp FROM submissions WHERE id = %s", (submission_id,))
    submission_data = cur.fetchone()

    if submission_data:
        participation_id = submission_data[0]
        task_id = submission_data[1]
        timestamp = submission_data[2]

        # 根據 participation_id 查詢 participations 資料，獲取 contest_id 和 user_id
        cur.execute("SELECT contest_id, user_id FROM participations WHERE id = %s", (participation_id,))
        participation_data = cur.fetchone()

        if participation_data:
            contest_id = participation_data[0]
            user_id = participation_data[1]

            # 根據 contest_id 查詢 contests 資料，獲取 start
            cur.execute("SELECT start FROM contests WHERE id = %s", (contest_id,))
            contest_data = cur.fetchone()

            contest_start = contest_data[0] if contest_data else "Unknown"

            # 根據 user_id 查詢 users 資料，獲取 first_name
            cur.execute("SELECT first_name FROM users WHERE id = %s", (user_id,))
            user_data = cur.fetchone()

            first_name = user_data[0] if user_data else "Unknown"

            # 檢查並更新該 user 和 contest 的最高分數
            if (user_id, contest_id) not in user_contest_scores:
                user_contest_scores[(user_id, contest_id)] = {
                    "first_name": first_name,
                    "contest_start": contest_start,
                    "total_score": score,
                    "timestamps": {task_id: timestamp}
                }
            else:
                if task_id not in user_contest_scores[(user_id, contest_id)]["timestamps"]:
                    user_contest_scores[(user_id, contest_id)]["timestamps"][task_id] = timestamp
                else:
                    current_timestamp = user_contest_scores[(user_id, contest_id)]["timestamps"][task_id]
                    if timestamp < current_timestamp:
                        user_contest_scores[(user_id, contest_id)]["timestamps"][task_id] = timestamp

                user_contest_scores[(user_id, contest_id)]["total_score"] += score

# 提取每個 user 和 contest_id 中的時間差
final_results = []

for (user_id, contest_id), data in user_contest_scores.items():
    latest_timestamp = max(data["timestamps"].values())
    contest_start = data["contest_start"]
    time_difference = latest_timestamp - contest_start
    total_seconds = int(time_difference.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_difference_str = f"{hours:02}:{minutes:02}:{seconds:02}"

    if data["first_name"].startswith("S"):
        final_results.append({
            "first_name": data["first_name"],
            "total_score": data["total_score"],
            "time_difference": time_difference_str
        })

# 合併相同 first_name 的資料
aggregated_results = {}

for result in final_results:
    first_name = result["first_name"]
    total_score = result["total_score"]

    time_parts = result["time_difference"].split(":")
    time_difference = timedelta(
        hours=int(time_parts[0]),
        minutes=int(time_parts[1]),
        seconds=int(time_parts[2])
    )

    if first_name not in aggregated_results:
        aggregated_results[first_name] = {
            "first_name": first_name,
            "total_score": total_score,
            "time_difference": time_difference
        }
    else:
        aggregated_results[first_name]["total_score"] += total_score
        aggregated_results[first_name]["time_difference"] += time_difference

# 格式化 time_difference 並轉為列表
final_aggregated_results = []

for data in aggregated_results.values():
    total_seconds = int(data["time_difference"].total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    data["time_difference"] = f"{hours:02}:{minutes:02}:{seconds:02}"
    final_aggregated_results.append(data)

final_aggregated_results = sorted(final_aggregated_results, key=lambda x: x["first_name"])

# 寫入 CSV 檔案
with open("output_aggregated.csv", "w", newline="") as csvfile:
    fieldnames = ["first_name", "total_score", "time_difference"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(final_aggregated_results)

# 關閉游標和資料庫連接
cur.close()
conn.close()

print("合併後的 CSV 檔案已生成：output_aggregated.csv")
