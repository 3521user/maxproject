from docx import Document

doc = Document('thesis_output.docx')
text = ''
for para in doc.paragraphs:
    text += para.text

total_chars = len(text)
chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')

print(f'总字符数（含空格）: {total_chars}')
print(f'估算中文字符数: {chinese_chars}')
print(f'估算总字数（中文*2 + 英文/2）: {int(chinese_chars + (total_chars - chinese_chars) / 2)}')