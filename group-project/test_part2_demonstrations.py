"""
Part II Task 2: 三个真实场景演示
每个任务要求：
- 至少 5 步
- 使用至少 3 个不同工具
- 记录完整的 Agent 轨迹
"""

import os
import sys
import json
from datetime import datetime, timedelta
from src.llm_client import DeepSeekClient
from src.personal_assistant_agent import PersonalAssistantAgent
from src.assistaant_tools.auth_helper import get_google_credentials
from src.utils import load_env

# 加载环境变量
load_env()

def print_separator(title=""):
    """打印分隔线"""
    if title:
        print(f"\n{'='*80}")
        print(f"  {title}")
        print(f"{'='*80}\n")
    else:
        print(f"\n{'-'*80}\n")

def print_step_info(step_num, tool_name, description):
    """打印步骤信息"""
    print(f"\n{'🔧 STEP ' + str(step_num):<70}")
    print(f"   Tool: {tool_name}")
    print(f"   Action: {description}")
    print(f"{'-'*80}")

def save_trajectory(task_name, trajectory, output_file="results/part2_trajectories.jsonl"):
    """保存轨迹到 JSONL 文件"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 添加任务名称标识
    trajectory_with_name = {
        "task_name": task_name,
        **trajectory
    }
    
    # 追加模式写入
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(trajectory_with_name, ensure_ascii=False) + "\n")
    
    print(f"\n✅ Trajectory saved to {output_file}")

def print_trajectory_summary(trajectory):
    """打印轨迹摘要（用于报告）"""
    print_separator("TRAJECTORY SUMMARY (FOR REPORT)")
    
    print(f"📝 Task Request:")
    print(f"   {trajectory['task']}\n")
    
    print(f"📊 Execution Statistics:")
    print(f"   • Total Steps: {len(trajectory['steps'])}")
    print(f"   • Tools Used: {len(trajectory['tools_used'])} different tools")
    print(f"   • Tool List: {', '.join(trajectory['tools_used'])}")
    print(f"   • Success: {'✅ YES' if trajectory['success'] else '❌ NO'}\n")
    
    print(f"🔍 Step-by-Step Breakdown:")
    for i, step in enumerate(trajectory['steps'], 1):
        print(f"\n   Step {i}: {step['tool']}")
        print(f"   ├─ Arguments: {json.dumps(step.get('arguments', {}), ensure_ascii=False)[:100]}...")
        if step.get('result'):
            result_str = str(step['result'])[:150]
            print(f"   ├─ Result: {result_str}...")
        print(f"   └─ Success: {'✅' if step.get('success') else '❌'}")
    
    print(f"\n📋 Final Summary:")
    print(f"   {trajectory['final_summary'][:500]}...")
    
    print_separator()

def task1_cycling_planning():
    """
    任务 1: 天气驱动的户外活动规划
    
    场景：安排周末户外骑行，需要检查天气、检查日历冲突、找租赁店、规划路线、创建日历事件
    预期步骤：5步
    预期工具：Weather (forecast) + Calendar (list, create) + Maps (search, directions)
    """
    print_separator("TASK 1: 天气驱动的户外活动规划 🚴")
    
    print("📖 Scenario:")
    print("   You want to plan a cycling trip this weekend, but only if the weather is good")
    print("   and you have no schedule conflicts. The agent should check weather, check calendar,")
    print("   find bike rental shops, get directions, and create a calendar event.\n")
    
    # 初始化 Agent
    llm_client = DeepSeekClient(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )
    google_creds = get_google_credentials()
    agent = PersonalAssistantAgent(
        llm_client=llm_client,
        google_credentials=google_creds,
        max_steps=15,
        verbose=True
    )
    
    # 计算这个周六的日期
    today = datetime.now()
    days_until_saturday = (5 - today.weekday()) % 7
    if days_until_saturday == 0:
        days_until_saturday = 7
    saturday = today + timedelta(days=days_until_saturday)
    saturday_str = saturday.strftime("%Y-%m-%d")
    
    # 用户请求
    user_request = f"""I want to plan a 2-hour cycling trip this Saturday ({saturday_str}) at 9am in Hong Kong.

Please help me organize this cycling trip. I need to make sure:
- The weather will be suitable for cycling (not raining, temperature between 15-28°C)
- I don't have any schedule conflicts on Saturday morning
- Find a good bike rental shop near Central Hong Kong
- Get directions on how to get there from Central MTR Station
- Create a calendar event with all the details

Only proceed if the weather is good and I don't have conflicts. If there are any issues, please let me know and suggest alternatives."""
    
    print(f"👤 User Request:")
    print(f"   {user_request}\n")
    
    print_separator("EXECUTING TASK")
    
    # 执行任务
    trajectory = agent.execute_task(user_request)
    
    # 保存轨迹
    save_trajectory("Task1_Cycling_Planning", trajectory)
    
    # 打印详细摘要
    print_trajectory_summary(trajectory)
    
    # 验证要求
    print_separator("REQUIREMENT VALIDATION")
    step_count = len(trajectory['steps'])
    tool_count = len(trajectory['tools_used'])
    
    print(f"✅ Minimum Steps (≥5): {'PASS ✓' if step_count >= 5 else f'FAIL ✗ (only {step_count} steps)'}")
    print(f"✅ Minimum Tools (≥3): {'PASS ✓' if tool_count >= 3 else f'FAIL ✗ (only {tool_count} tools)'}")
    print(f"✅ Task Success: {'PASS ✓' if trajectory['success'] else 'FAIL ✗'}")
    
    return trajectory

def task2_lunch_meeting():
    """
    任务 2: 会议安排与地点推荐
    
    场景：安排明天午餐会议，需要检查日历冲突、找餐厅、获取路线、创建事件
    预期步骤：7-8步
    预期工具：Calendar (list, create) + Maps (search, directions) + Weather (current)
    """
    print_separator("TASK 2: 会议安排与地点推荐 🍽️")
    
    print("📖 Scenario:")
    print("   You need to schedule a lunch meeting tomorrow at an Italian restaurant near HKUST.")
    print("   The agent should check for schedule conflicts, find a good restaurant,")
    print("   get directions, check weather, and create the calendar event.\n")
    
    # 初始化 Agent
    llm_client = DeepSeekClient(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )
    google_creds = get_google_credentials()
    agent = PersonalAssistantAgent(
        llm_client=llm_client,
        google_credentials=google_creds,
        max_steps=15,
        verbose=True
    )
    
    # 计算明天的日期
    tomorrow = datetime.now() + timedelta(days=1)
    tomorrow_str = tomorrow.strftime("%Y-%m-%d")
    
    # 用户请求
    user_request = f"""I need to schedule a lunch meeting tomorrow ({tomorrow_str}) at 12:00 PM at an Italian restaurant near HKUST.

Please help me arrange this lunch meeting. I need you to:
- Check if I have any schedule conflicts tomorrow around lunch time (11am-2pm)
- Find a good Italian restaurant near Hong Kong University of Science and Technology (prefer highly rated ones)
- Get walking directions from HKUST to the restaurant
- Check the current weather so I know if we'll need an umbrella
- Create a calendar event called "Lunch Meeting - Italian Restaurant" from 12:00 PM to 1:30 PM with all the details

If there's a conflict in my schedule, please suggest alternative times."""
    
    print(f"👤 User Request:")
    print(f"   {user_request}\n")
    
    print_separator("EXECUTING TASK")
    
    # 执行任务
    trajectory = agent.execute_task(user_request)
    
    # 保存轨迹
    save_trajectory("Task2_Lunch_Meeting", trajectory)
    
    # 打印详细摘要
    print_trajectory_summary(trajectory)
    
    # 验证要求
    print_separator("REQUIREMENT VALIDATION")
    step_count = len(trajectory['steps'])
    tool_count = len(trajectory['tools_used'])
    
    print(f"✅ Minimum Steps (≥5): {'PASS ✓' if step_count >= 5 else f'FAIL ✗ (only {step_count} steps)'}")
    print(f"✅ Minimum Tools (≥3): {'PASS ✓' if tool_count >= 3 else f'FAIL ✗ (only {tool_count} tools)'}")
    print(f"✅ Task Success: {'PASS ✓' if trajectory['success'] else 'FAIL ✗'}")
    
    return trajectory

def task3_travel_preparation():
    """
    任务 3: 旅行准备与提醒设置
    
    场景：准备下周的东京旅行，需要查天气、搜索打包清单、查交通、创建多个提醒
    预期步骤：8-9步
    预期工具：Google Search + Weather (forecast) + Maps (directions) + Calendar (create x2)
    """
    print_separator("TASK 3: 旅行准备与提醒设置 ✈️")
    
    print("📖 Scenario:")
    print("   You're traveling to Tokyo next week and need to prepare.")
    print("   The agent should check Tokyo weather, search packing tips, find airport transport,")
    print("   and create multiple calendar reminders.\n")
    
    # 初始化 Agent
    llm_client = DeepSeekClient(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )
    google_creds = get_google_credentials()
    agent = PersonalAssistantAgent(
        llm_client=llm_client,
        google_credentials=google_creds,
        max_steps=20,
        verbose=True
    )
    
    # 计算下周的日期（假设7天后出发）
    departure_date = datetime.now() + timedelta(days=7)
    departure_str = departure_date.strftime("%Y-%m-%d")
    
    packing_date = departure_date - timedelta(days=2)  # 提前2天打包
    packing_str = packing_date.strftime("%Y-%m-%d")
    
    # 用户请求
    user_request = f"""I'm traveling to Tokyo on {departure_str} and need help preparing for the trip.

Please help me get ready for this trip. I need to:
- Understand what the weather will be like in Tokyo so I know what to pack
- Find out what essentials I should bring for a Tokyo winter trip
- Know how to get from Narita Airport to Shibuya Station when I arrive
- Compare Tokyo's weather with Hong Kong's current weather
- Set up a packing reminder for {packing_str} at 7:00 PM (2 days before departure, 2 hours duration)
- Create a departure reminder for {departure_str} at 8:00 AM at Hong Kong International Airport (2 hours duration)

Please include all the relevant travel information (weather, packing tips, transportation options) in the calendar event descriptions so I have everything I need."""
    
    print(f"👤 User Request:")
    print(f"   {user_request}\n")
    
    print_separator("EXECUTING TASK")
    
    # 执行任务
    trajectory = agent.execute_task(user_request)
    
    # 保存轨迹
    save_trajectory("Task3_Travel_Preparation", trajectory)
    
    # 打印详细摘要
    print_trajectory_summary(trajectory)
    
    # 验证要求
    print_separator("REQUIREMENT VALIDATION")
    step_count = len(trajectory['steps'])
    tool_count = len(trajectory['tools_used'])
    
    print(f"✅ Minimum Steps (≥5): {'PASS ✓' if step_count >= 5 else f'FAIL ✗ (only {step_count} steps)'}")
    print(f"✅ Minimum Tools (≥3): {'PASS ✓' if tool_count >= 3 else f'FAIL ✗ (only {tool_count} tools)'}")
    print(f"✅ Task Success: {'PASS ✓' if trajectory['success'] else 'FAIL ✗'}")
    
    return trajectory

def main():
    """运行所有三个演示任务"""
    print_separator("PART II TASK 2: THREE REALISTIC DEMONSTRATIONS")
    print("Running three complex tasks to demonstrate the Personal Assistant Agent")
    print("Each task must have ≥5 steps and use ≥3 different tools\n")
    
    # 清空之前的轨迹文件
    output_file = "results/part2_trajectories.jsonl"
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"🗑️  Cleared previous trajectories from {output_file}\n")
    
    # 运行三个任务
    trajectories = []
    
    try:
        print("\n" + "="*80)
        print("STARTING TASK 1 OF 3")
        print("="*80)
        traj1 = task1_cycling_planning()
        trajectories.append(traj1)
        input("\n⏸️  Press Enter to continue to Task 2...")
        
    except Exception as e:
        print(f"\n❌ Task 1 failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    try:
        print("\n" + "="*80)
        print("STARTING TASK 2 OF 3")
        print("="*80)
        traj2 = task2_lunch_meeting()
        trajectories.append(traj2)
        input("\n⏸️  Press Enter to continue to Task 3...")
        
    except Exception as e:
        print(f"\n❌ Task 2 failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    try:
        print("\n" + "="*80)
        print("STARTING TASK 3 OF 3")
        print("="*80)
        traj3 = task3_travel_preparation()
        trajectories.append(traj3)
        
    except Exception as e:
        print(f"\n❌ Task 3 failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # 最终总结
    print_separator("FINAL SUMMARY - ALL THREE TASKS")
    
    for i, traj in enumerate(trajectories, 1):
        print(f"\n📊 Task {i}: {traj.get('task_name', f'Task {i}')}")
        print(f"   • Steps: {len(traj['steps'])} {'✓' if len(traj['steps']) >= 5 else '✗ (< 5)'}")
        print(f"   • Tools: {len(traj['tools_used'])} {'✓' if len(traj['tools_used']) >= 3 else '✗ (< 3)'}")
        print(f"   • Success: {'✓' if traj['success'] else '✗'}")
        print(f"   • Tool List: {', '.join(traj['tools_used'])}")
    
    print(f"\n{'='*80}")
    print(f"✅ All trajectories saved to: {output_file}")
    print(f"📄 You can now copy the trajectories to your report!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
