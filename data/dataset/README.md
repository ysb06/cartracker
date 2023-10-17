# Songdo Dataset

## 개요

2022년 10월 5일부터 2022년 10월 7일까지 SCIGC 소속 실험차가 나타난 동영상과 실험차에 Bounding Box를 표시한 데이터셋입니다. 실험차의 Object Detection 모델 생성을 목적으로 만들어졌습니다.

## 데이터 구조
- label.json

  Label Studio를 통해 Bouding Box 표시한 라벨 데이터가 포함되어 있음. Label Studio Common Format 형식으로 자세한 데이터 구조는 Label Studio 홈페이지 참조

- info.yaml

  Label Studio와 별도로 생성한 정보 파일. 내부 파일들을 Dataset으로 구성한 시점의 정보를 나타내고 있다. Label Studio에서 작업한 비디오 파일명과 현재 폴더의 파일명에 차이가 있으므로 Label의 각 Task가 어떤 파일에 연계되어 있는지에 대한 정보도 포함되어 있음. (task_related_video_filename_list)

- 레이블 대상 파일들

  기본적으로 Label Studio에서 작업 가능한 webm형식의 비디오 파일로 구성됨

## 참고

레이블 데이터 불러오는 것은 cartracker 프로젝트의 dataset 패키지 참조

  