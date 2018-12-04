# -*- coding:utf-8 -*-
#爬取图片
import urllib.request
import time
import re
def generate_allurl(user_in_nub):
    #url = 'http://skqs.guoxuedashi.com/wen_580f/124{}.html'
    url = 'https://www.bilibili.com/video/av15200856/?p=2'
    for url_next in range(2, int(user_in_nub)):
        yield url.format(url_next)
def main():
     all_url = []
     user_in_nub = input('输入生成页数：')
     for i in generate_allurl(user_in_nub):
          all_url.append(i)
     return all_url

if __name__ == '__main__':
     all_url = main()
print(all_url)
cnt=1
for i in range(len(all_url)):
     url = all_url[i]
     print(url)
     try:
          response = urllib.request.urlopen(url,timeout = 500)
          #print(response.getcode())
          html = response.read()
          # 提取其中所有的图片url(使用正则)
          #<img src="http://siku.guoxuedashi.com/wyg/WYG0364/WYG0364-0117a.png" style="max-width: 90%; max-height: 90%;">
          reg = r'(http://.*?\.png)'
          imgre = re.compile(reg)
          html = html.decode('utf-8')
          imlist = re.findall(reg, html)
          response.close()
     except urllib.error.URLError as e:
          print(e.reason)
     time.sleep(1)
     for imurl in imlist:
          print (cnt)
          urllib.request.urlretrieve(imurl, "%s.jpg" % cnt);
          cnt+=1
