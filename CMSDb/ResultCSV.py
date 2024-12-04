import psycopg2
import csv
from datetime import datetime, timedelta

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

# 用來追蹤每個 user 和 contest_id 的最高分數總和
user_contest_scores = {}

# 查詢 submission_results 資料，根據 submission_id 得到 score 和 submission_id
cur.execute("SELECT submission_id, score FROM submission_results")
submission_results = cur.fetchall()

for submission_result in submission_results:
    submission_id = submission_result[0]  # 提取 submission_id
    score = submission_result[1]  # 提取 score

    # 根據 submission_id 查詢 submissions 資料，得到 participation_id, task_id, timestamp
    cur.execute("SELECT participation_id, task_id, timestamp FROM submissions WHERE id = %s", (submission_id,))
    submission_data = cur.fetchone()

    if submission_data:
        participation_id = submission_data[0]
        task_id = submission_data[1]
        timestamp = submission_data[2]

        # 根據 participation_id 查詢 participations 資料，得到 contest_id 和 user_id
        cur.execute("SELECT contest_id, user_id FROM participations WHERE id = %s", (participation_id,))
        participation_data = cur.fetchone()

        if participation_data:
            contest_id = participation_data[0]
            user_id = participation_data[1]

            # 根據 contest_id 查詢 contests 資料，得到 contest_name 和 start
            cur.execute("SELECT name, start FROM contests WHERE id = %s", (contest_id,))
            contest_data = cur.fetchone()

            if contest_data:
                contest_name = contest_data[0]
                contest_start = contest_data[1]
            else:
                contest_name = "Unknown"
                contest_start = "Unknown"

            # 根據 user_id 查詢 users 資料，得到 first_name
            cur.execute("SELECT first_name FROM users WHERE id = %s", (user_id,))
            user_data = cur.fetchone()

            if user_data:
                first_name = user_data[0]
            else:
                first_name = "Unknown"

            if (user_id, contest_id) not in user_contest_scores:
                user_contest_scores[(user_id, contest_id)] = {
                    "first_name": first_name,
                    "contest_name": contest_name,
                    "contest_start": contest_start,
                    "total_score": score,
                    "task_scores": {task_id: score},
                    "timestamps": {task_id: timestamp}
                }
            else:
                if task_id not in user_contest_scores[(user_id, contest_id)]["task_scores"]:
                    user_contest_scores[(user_id, contest_id)]["task_scores"][task_id] = score
                    user_contest_scores[(user_id, contest_id)]["timestamps"][task_id] = timestamp
                else:
                    current_score = user_contest_scores[(user_id, contest_id)]["task_scores"][task_id]
                    current_timestamp = user_contest_scores[(user_id, contest_id)]["timestamps"][task_id]

                    if score > current_score:
                        user_contest_scores[(user_id, contest_id)]["task_scores"][task_id] = score
                        user_contest_scores[(user_id, contest_id)]["timestamps"][task_id] = timestamp
                    elif score == current_score and timestamp < current_timestamp:
                        user_contest_scores[(user_id, contest_id)]["timestamps"][task_id] = timestamp

                user_contest_scores[(user_id, contest_id)]["total_score"] = sum(
                    user_contest_scores[(user_id, contest_id)]["task_scores"].values()
                )

# 提取每個 user 和 contest_id 中的最新 timestamp
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
            "contest_name": data["contest_name"],
            "contest_start": contest_start.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp": latest_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "total_score": data["total_score"],
            "time_difference": time_difference_str
        })

# 整合相同使用者的結果
aggregated_results = {}

for result in final_results:
    first_name = result["first_name"]
    time_difference = result["time_difference"]
    total_score = result["total_score"]

    # 轉換 time_difference 為秒數
    time_parts = list(map(int, time_difference.split(":")))
    time_difference_seconds = time_parts[0] * 3600 + time_parts[1] * 60 + time_parts[2]

    if first_name not in aggregated_results:
        aggregated_results[first_name] = {
            "first_name": first_name,
            "total_score": total_score,
            "time_difference_seconds": time_difference_seconds
        }
    else:
        aggregated_results[first_name]["total_score"] += total_score
        aggregated_results[first_name]["time_difference_seconds"] += time_difference_seconds

final_aggregated_results = []
for data in aggregated_results.values():
    total_seconds = data["time_difference_seconds"]
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_difference_str = f"{hours:02}:{minutes:02}:{seconds:02}"

    final_aggregated_results.append({
        "first_name": data["first_name"],
        "total_score": data["total_score"],
        "time_difference": time_difference_str
    })

# 按照 first_name 排序
final_aggregated_results_sorted = sorted(final_aggregated_results, key=lambda x: x["first_name"])

with open("All_Results.csv", "w", newline="") as csvfile:
    fieldnames = ["first_name", "total_score", "time_difference"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for result in final_aggregated_results_sorted:
        writer.writerow(result)

cur.close()
conn.close()

print(final_aggregated_results_sorted)
print("整合後的 CSV 檔案已生成：All_Results.csv")
