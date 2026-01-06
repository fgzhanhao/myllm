"""下载训练数据"""
import urllib.request
import os

DATA_DIR = 'data'

def download_chinese():
    """下载中文预训练数据（古诗词+成语+百科）"""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 中文诗词和常识文本
    chinese_text = """
什么是人工智能？人工智能是让计算机模拟人类智能的技术。
中国的首都是哪里？中国的首都是北京。
太阳从哪边升起？太阳从东边升起。
一年有多少天？一年有三百六十五天。
水的化学式是什么？水的化学式是H2O。
地球是什么形状？地球是球形的。
谁发明了电灯？爱迪生发明了电灯。
长城在哪个国家？长城在中国。
月亮绕着什么转？月亮绕着地球转。
人有多少颗牙齿？成年人有三十二颗牙齿。

床前明月光，疑是地上霜。举头望明月，低头思故乡。
春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。
白日依山尽，黄河入海流。欲穷千里目，更上一层楼。
锄禾日当午，汗滴禾下土。谁知盘中餐，粒粒皆辛苦。

天地玄黄，宇宙洪荒。日月盈昃，辰宿列张。
寒来暑往，秋收冬藏。闰余成岁，律吕调阳。

学而时习之，不亦说乎。有朋自远方来，不亦乐乎。
知之为知之，不知为不知，是知也。
三人行，必有我师焉。择其善者而从之，其不善者而改之。
"""
    
    filepath = os.path.join(DATA_DIR, 'chinese.txt')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(chinese_text * 100)  # 重复增加数据量
    
    size = os.path.getsize(filepath) / 1024
    print(f"中文数据已创建: {filepath} ({size:.1f} KB)")
    return filepath


def download_qa():
    """创建问答训练数据"""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    qa_pairs = """<问>什么是人工智能？<答>人工智能是让计算机模拟人类智能的技术。
<问>中国的首都是哪里？<答>中国的首都是北京。
<问>太阳从哪边升起？<答>太阳从东边升起。
<问>一年有多少天？<答>一年有三百六十五天。
<问>水的化学式是什么？<答>水的化学式是H2O。
<问>地球是什么形状？<答>地球是球形的。
<问>谁发明了电灯？<答>爱迪生发明了电灯。
<问>长城在哪个国家？<答>长城在中国。
<问>月亮绕着什么转？<答>月亮绕着地球转。
<问>人有多少颗牙齿？<答>成年人有三十二颗牙齿。
<问>一周有几天？<答>一周有七天。
<问>一天有多少小时？<答>一天有二十四小时。
<问>中国最长的河流是什么？<答>中国最长的河流是长江。
<问>世界上最高的山是什么？<答>世界上最高的山是珠穆朗玛峰。
<问>火是什么颜色？<答>火通常是红色或橙色的。
<问>冰是什么状态的水？<答>冰是固态的水。
<问>植物需要什么才能生长？<答>植物需要阳光、水和空气才能生长。
<问>鱼用什么呼吸？<答>鱼用鳃呼吸。
<问>人用什么呼吸？<答>人用肺呼吸。
<问>天空是什么颜色？<答>天空通常是蓝色的。
"""
    
    filepath = os.path.join(DATA_DIR, 'qa.txt')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(qa_pairs * 50)  # 重复增加数据量
    
    size = os.path.getsize(filepath) / 1024
    print(f"问答数据已创建: {filepath} ({size:.1f} KB)")
    return filepath


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("用法:")
        print("  python download_data.py chinese  # 中文预训练数据")
        print("  python download_data.py qa       # 问答数据")
    else:
        cmd = sys.argv[1]
        if cmd == 'chinese':
            download_chinese()
        elif cmd == 'qa':
            download_qa()
        else:
            print(f"未知命令: {cmd}")
