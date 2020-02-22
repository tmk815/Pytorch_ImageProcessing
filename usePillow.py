from PIL import Image
import matplotlib.pyplot as plt

# 画像取得
image = Image.open('content/owl.jpg')
plt.imshow(image)  # 描画

# グレースケール化
gray_img = image.convert('L')
plt.imshow(gray_img)  # 描画
plt.show()
