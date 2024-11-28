import psycopg2
import csv
from datetime import datetime

# 連接資料庫
conn = psycopg2.connect(
    dbname="cmsdb",  
    user="cmsuser",            
    password="cmsuser",     
    host="localhost",           
    port="5432"                   
)

cur = conn.cursor()

# 查詢 users 資料
cur.execute("SELECT id, first_name FROM users")
users_data = cur.fetchall()

# 用於記錄已經處理過的 first_name
processed_names = set()

# 開啟 CSV 檔案以便寫入
with open('output.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['first_name', 'user_id', 'contest_id', 'task_id', 'highest_score', 'timestamp']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    # 寫入 CSV 檔案的標題
    writer.writeheader()

    for result in users_data:
        user_id = result[0]
        user_name = result[1]
        
        # 只有第一次遇到該 first_name 時才處理
        if user_name not in processed_names:
            processed_names.add(user_name)  # 標記該 first_name 為已處理
            
            # 查詢 participations 資料，並同時獲取 contest_id
            cur.execute("SELECT id, contest_id FROM participations WHERE user_id = %s", (user_id,))
            participation_id_result = cur.fetchall()

            if participation_id_result:  # 確保參與記錄存在
                # 記錄每次遇到 task_id 時的最高分數
                highest_scores = {}  # 用來記錄每個 task_id 的最高分數

                for result2 in participation_id_result:
                    participation_id = result2[0]  # 提取單個 participation_id
                    contest_id = result2[1]  # 提取 contest_id
                    
                    # 查詢 submissions 資料，同時獲取 submission_id 和 task_id
                    cur.execute("SELECT id, task_id FROM submissions WHERE participation_id = %s", (participation_id,))
                    submission_id_result = cur.fetchall()

                    if submission_id_result:  # 如果有對應的 submission_id
                        for submission in submission_id_result:
                            submission_id = submission[0]  # 提取 submission_id
                            task_id = submission[1]  # 提取 task_id

                            # 查詢該 submission_id 的 score
                            cur.execute("SELECT score FROM submission_results WHERE submission_id = %s", (submission_id,))
                            score_result = cur.fetchone()
                            if score_result:
                                score = score_result[0]
                                
                                if score == 0.0:
                                    continue

                                # 每次遇到不同的 task_id 時，重設該 task_id 的最高分數
                                if task_id not in highest_scores:
                                    highest_scores[task_id] = score
                                else:
                                    # 更新最高分數（如果當前分數更高）
                                    if score > highest_scores[task_id]:
                                        highest_scores[task_id] = score

                # 將結果寫入 CSV
                for task_id, highest_score in highest_scores.items():
                    # 查詢該 task_id 對應的 timestamp
                    cur.execute("SELECT timestamp FROM submissions WHERE task_id = %s AND participation_id IN (SELECT id FROM participations WHERE user_id = %s)", (task_id, user_id))
                    timestamp_result = cur.fetchone()
                    
                    if timestamp_result:
                        timestamp = timestamp_result[0]
                        
                        # 格式化時間戳，確保寫入 CSV 時正確顯示
                        if isinstance(timestamp, datetime):
                            timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')  # 格式化時間戳為字串

                        writer.writerow({
                            'first_name': user_name,
                            'user_id': user_id,
                            'contest_id': contest_id,
                            'task_id': task_id,
                            'highest_score': highest_score,
                            'timestamp': timestamp
                        })
                    else:
                        # 如果沒有找到 timestamp，可以選擇填入空值或其他標識
                        writer.writerow({
                            'first_name': user_name,
                            'user_id': user_id,
                            'contest_id': contest_id,
                            'task_id': task_id,
                            'highest_score': highest_score,
                            'timestamp': 'not found'
                        })
            else:
                print(f"participation_id not found for user_id: {user_id}")

# 關閉游標和資料庫連接
cur.close()
conn.close()
