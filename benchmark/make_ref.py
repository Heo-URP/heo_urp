import cv2
import os
import json


######### 경로 설정 #########
image_dir = "./benchmark/data/images"
output_file = "./benchmark/data/annotations.jsonl"




image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))])

with open(output_file, "w") as out_f:
    for img_name in image_files:
        img_path = os.path.join(image_dir, img_name)
        print(f"\n이미지: {img_path}")

        img = cv2.imread(img_path)
        if img is None:
            print(f"이미지 로드 실패: {img_path}")
            continue

        # 마우스로 바운딩박스 선택
        bbox = cv2.selectROI(f"Select ROI - {img_name}", img, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()

        if bbox == (0, 0, 0, 0):
            print("Error: 바운딩박스를 지정하지 않았습니다.")
            continue

        # Annotation 입력 (위치표현)
        ref_text = input("해당 객체를 설명하는 문장을 입력하세요: ").strip()


        data = {
            "image_path": os.path.join(image_dir, img_name),
            "image_id": os.path.splitext(img_name)[0],
            "text": ref_text,
            "bbox": [int(b) for b in bbox]  # x, y, w, h
        }

        # 저장
        out_f.write(json.dumps(data, ensure_ascii=False) + "\n")
        print("저장 완료:", data)
