import os
from PIL import Image

def merge_images():
    # 文件夹路径
    input_folder = 'generated_outputs'
    # 输出文件名
    output_filename = 'combined_2x5_grid.png'
    
    # 检查文件夹是否存在
    if not os.path.exists(input_folder):
        print(f"错误: 找不到文件夹 '{input_folder}'")
        return

    images = []
    # 按照数字 0 到 9 的顺序读取图片
    for x in range(10):
        filename = f"digit_{x}_grid.png"
        filepath = os.path.join(input_folder, filename)
        
        if os.path.exists(filepath):
            try:
                img = Image.open(filepath)
                images.append(img)
            except Exception as e:
                print(f"无法加载图片 {filepath}: {e}")
                return
        else:
            print(f"警告: 找不到图片 {filepath}")
            return

    if not images:
        print("没有加载到任何图片。")
        return

    # 假设所有图片尺寸相同，获取第一张图片的宽高
    width, height = images[0].size
    
    # 创建新的空白图，尺寸为 5列 * 宽，2行 * 高
    # 2 * 5 grid
    cols = 5
    rows = 2
    combined_image = Image.new('RGB', (width * cols, height * rows))

    # 将图片粘贴到新图中
    for index, img in enumerate(images):
        # 计算当前图片所在的行列 (row, col)
        # index 0-4 -> row 0
        # index 5-9 -> row 1
        row = index // cols
        col = index % cols
        
        # 计算粘贴位置坐标
        x_offset = col * width
        y_offset = row * height
        
        combined_image.paste(img, (x_offset, y_offset))

    # 保存结果
    combined_image.save(output_filename)
    print(f"拼接完成！图片已保存为: {output_filename}")

if __name__ == "__main__":
    merge_images()