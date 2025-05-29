import cv2
import os
import json


######### 경로 설정 #########
image_dir = "./data/images"
output_file = "./data/annotations.jsonl"




image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))])

with open(output_file, "a", encoding='utf-8') as out_f:
    for img_idx, img_name in enumerate(image_files):
        img_path = os.path.join(image_dir, img_name)
        print(f"\n{'='*50}")
        print(f"이미지 {img_idx + 1}/{len(image_files)}: {img_path}")
        print(f"{'='*50}")

        img = cv2.imread(img_path)
        if img is None:
            print(f"이미지 로드 실패: {img_path}")
            continue

        # 각 이미지당 3개의 바운딩박스 선택
        bboxes_data = []
        
        for bbox_num in range(3):
            print(f"\n--- {bbox_num + 1}번째 바운딩박스 선택 ---")
            print("마우스로 드래그하여 바운딩박스를 선택하세요.")
            print("선택 후 SPACE나 ENTER를 누르세요. 취소하려면 'c'를 누르세요.")
            
            # 마우스로 바운딩박스 선택
            bbox = cv2.selectROI(f"Select ROI {bbox_num + 1}/3 - {img_name}", img, fromCenter=False, showCrosshair=True)
            cv2.destroyAllWindows()

            if bbox == (0, 0, 0, 0):
                print(f"바운딩박스 {bbox_num + 1}를 선택하지 않았습니다.")
                
                # 사용자에게 선택지 제공
                choice = input("다시 시도하시겠습니까? (y: 재시도, s: 건너뛰기, q: 종료): ").strip().lower()
                
                if choice == 'y':
                    bbox_num -= 1  # 다시 시도
                    continue
                elif choice == 's':
                    print(f"{bbox_num + 1}번째 바운딩박스를 건너뜁니다.")
                    continue
                elif choice == 'q':
                    print("프로그램을 종료합니다.")
                    exit()
                else:
                    continue

            # Annotation 입력 (위치표현)
            while True:
                ref_text = input(f"{bbox_num + 1}번째 객체를 설명하는 문장을 입력하세요: ").strip()
                if ref_text:
                    break
                print("설명을 입력해주세요!")

            # 바운딩박스 데이터 저장
            data = {
                "image_path": img_path,
                "image_id": f"{os.path.splitext(img_name)[0]}_{bbox_num + 1:02d}",  # 예: 000000000776_01
                "original_image_id": os.path.splitext(img_name)[0],
                "bbox_number": bbox_num + 1,
                "text": ref_text,
                "bbox": [int(b) for b in bbox]  # x, y, w, h
            }
            
            bboxes_data.append(data)
            print(f"바운딩박스 {bbox_num + 1} 저장 완료: {data['text']}")

        # 선택된 모든 바운딩박스를 파일에 저장
        if bboxes_data:
            for data in bboxes_data:
                out_f.write(json.dumps(data, ensure_ascii=False) + "\n")
            out_f.flush()  # 즉시 파일에 쓰기
            
            print(f"\n✅ {img_name}에 대한 {len(bboxes_data)}개 어노테이션 저장 완료!")
        else:
            print(f"\n❌ {img_name}에 대한 어노테이션이 없습니다.")
        
        # 다음 이미지로 진행 여부 확인 (마지막 이미지가 아닌 경우)
        if img_idx < len(image_files) - 1:
            continue_choice = input(f"\n다음 이미지로 진행하시겠습니까? (y/n): ").strip().lower()
            if continue_choice == 'n':
                print("작업을 중단합니다.")
                break

print(f"\n🎉 모든 작업 완료! 결과 파일: {output_file}")

# 결과 요약 출력
if os.path.exists(output_file):
    with open(output_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print(f"총 {len(lines)}개의 어노테이션이 생성되었습니다.")