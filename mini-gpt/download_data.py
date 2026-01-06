"""
下载训练数据脚本
支持下载莎士比亚全集和中文小说
"""
import urllib.request
import os

DATA_DIR = 'data'

DATASETS = {
    'shakespeare': {
        'url': 'https://www.gutenberg.org/cache/epub/100/pg100.txt',
        'file': 'shakespeare.txt',
        'desc': '莎士比亚全集 (~5.5MB)',
    },
    'tiny_shakespeare': {
        'url': 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
        'file': 'tiny_shakespeare.txt',
        'desc': '莎士比亚精选 (~1MB)',
    },
}

def download(name):
    """下载指定数据集"""
    if name not in DATASETS:
        print(f"未知数据集: {name}")
        print(f"可用: {list(DATASETS.keys())}")
        return
    
    info = DATASETS[name]
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, info['file'])
    
    if os.path.exists(filepath):
        print(f"文件已存在: {filepath}")
        return filepath
    
    print(f"正在下载 {info['desc']}...")
    print(f"URL: {info['url']}")
    
    try:
        urllib.request.urlretrieve(info['url'], filepath)
        size = os.path.getsize(filepath) / 1024 / 1024
        print(f"下载完成: {filepath} ({size:.2f} MB)")
        return filepath
    except Exception as e:
        print(f"下载失败: {e}")
        return None


def create_chinese_sample():
    """创建中文示例数据（用于测试中文训练）"""
    chinese_text = """
天地玄黄，宇宙洪荒。日月盈昃，辰宿列张。
寒来暑往，秋收冬藏。闰余成岁，律吕调阳。
云腾致雨，露结为霜。金生丽水，玉出昆冈。
剑号巨阙，珠称夜光。果珍李柰，菜重芥姜。
海咸河淡，鳞潜羽翔。龙师火帝，鸟官人皇。
始制文字，乃服衣裳。推位让国，有虞陶唐。
吊民伐罪，周发殷汤。坐朝问道，垂拱平章。

床前明月光，疑是地上霜。举头望明月，低头思故乡。
春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。
白日依山尽，黄河入海流。欲穷千里目，更上一层楼。
红豆生南国，春来发几枝。愿君多采撷，此物最相思。

君不见黄河之水天上来，奔流到海不复回。
君不见高堂明镜悲白发，朝如青丝暮成雪。
人生得意须尽欢，莫使金樽空对月。
天生我材必有用，千金散尽还复来。

大江东去，浪淘尽，千古风流人物。
故垒西边，人道是，三国周郎赤壁。
乱石穿空，惊涛拍岸，卷起千堆雪。
江山如画，一时多少豪杰。

遥想公瑾当年，小乔初嫁了，雄姿英发。
羽扇纶巾，谈笑间，樯橹灰飞烟灭。
故国神游，多情应笑我，早生华发。
人生如梦，一尊还酹江月。
"""
    
    filepath = os.path.join(DATA_DIR, 'chinese_poems.txt')
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 重复多次增加数据量
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(chinese_text * 50)
    
    size = os.path.getsize(filepath) / 1024
    print(f"中文诗词数据已创建: {filepath} ({size:.1f} KB)")
    return filepath


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  python download_data.py shakespeare    # 下载莎士比亚全集")
        print("  python download_data.py tiny           # 下载精选版")
        print("  python download_data.py chinese        # 创建中文诗词数据")
        print("\n可用数据集:")
        for name, info in DATASETS.items():
            print(f"  {name}: {info['desc']}")
    else:
        name = sys.argv[1]
        if name == 'chinese':
            create_chinese_sample()
        elif name == 'tiny':
            download('tiny_shakespeare')
        else:
            download(name)
