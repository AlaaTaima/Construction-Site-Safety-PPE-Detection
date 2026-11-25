# حارس الموقع (Site Guardian)

تطبيق ويب لكشف معدات السلامة (الخوذة، السترة، الكمامة) باستخدام YOLOv8 المصدّر بصيغة ONNX.
الكاميرا تُفتح من المتصفح وتُرسل الإطارات للخادم، لذلك يعمل على Render بدون كاميرا على السيرفر.

## الملفات
- app.py: الخادم (Flask) والاستدلال بـ ONNX Runtime ومنطق الالتزام
- static/index.html: الواجهة
- models/best.onnx: ضع النموذج هنا (من Colab)
- requirements.txt و .python-version: متطلبات النشر

## تشغيل محلي
pip install -r requirements.txt
python app.py        # ثم افتح http://localhost:5000

## النشر على Render
1. ارفع هذا المجلد إلى GitHub (مع models/best.onnx).
2. Render: New + ثم Web Service ثم اختر الريبو.
3. Root Directory: render-app (أو اتركه فارغاً إذا كان هذا المجلد هو جذر الريبو)
4. Runtime: Python 3
5. Build Command:  pip install -r requirements.txt
6. Start Command:  gunicorn app:app --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT
7. Instance Type: Free ثم Create Web Service
8. بعد النشر افتح الرابط (HTTPS مطلوب لتشغيل الكاميرا).

## ملاحظة عن ONNX
أحدث إصدارات ultralytics تصدّر ONNX بإصدار opset 22 أو أعلى، وهذا يتطلب onnxruntime حديث (مثبّت في requirements.txt).
بعد التصدير تأكد أن سجل التصدير يطبع: output shape(s) (1, 14, 8400)
