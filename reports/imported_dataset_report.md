# Imported Dataset Report

Generated: 2026-06-29T21:40:23

## Target YOLO Classes

| id | name |
| --- | --- |
| 0 | left |
| 1 | right |
| 2 | turnaround |
| 3 | park |
| 4 | person |
| 5 | obstacle |
| 6 | crosswalk |

## YOLO Summary

- accepted images: 5057
- empty labels after mapping: 0
- split counts: test:507, train:4045, val:505
- corrected park relabel: matched 471, kept 445, removed empty 26
- class distribution: 0:left=1536, 1:right=1470, 2:turnaround=1070, 3:park=409, 4:person=0, 5:obstacle=1535, 6:crosswalk=762

| source | seen labels | accepted images | skipped | source classes | skipped objects |
| --- | ---: | ---: | --- | --- | --- |
| g01_yolo_06 | 150 | 150 | - | left, right, return, obs1, obs2, leftdw | - |
| g01_yolo_07 | 150 | 149 | empty_after_mapping:1 | left, right, return, obs1, obs2, leftdw | - |
| g01_yolo_08 | 100 | 100 | - | left, right, return, obs1, obs2, leftdw | - |
| g01_yolo_09 | 100 | 100 | - | left, right, return, obs1, obs2, leftdw | - |
| g01_yolo_10 | 735 | 735 | - | left, right, return, obs1, obs2, leftdw | - |
| g01_yolo_11 | 178 | 178 | - | left, right, return, obs1, obs2, leftdw | - |
| g01_yolo_12 | 108 | 108 | - | left, right, return, obs1, obs2, leftdw | - |
| g01_yolo_13 | 162 | 162 | - | left, right, return, obs1, obs2, leftdw | - |
| g01_yolo_14 | 93 | 91 | missing_image:2 | left, right, return, obs1, obs2, leftdw | - |
| g01_yolo_15 | 802 | 801 | missing_image:1 | left, right, return, obs1, obs2, leftdw | - |
| g04_landmark | 1250 | 957 | empty_after_mapping:43, missing_image:250, removed_by_park_relabel:20 | park, stop, right, back, sideway, left | stop:113 |
| g16_yolo_road1 | 598 | 598 | removed_by_park_relabel:3 | turnaround, crosswalk, left, right, park | - |
| g16_yolo_road2 | 427 | 427 | - | turnaround, crosswalk, left, right, park | - |
| g23_sign | 644 | 527 | empty_after_mapping:117, removed_by_park_relabel:3 | park, stop, right, back, sideway, left | stop:126 |

YOLO class remapping used:

- `back` -> 2:turnaround
- `crosswalk` -> 6:crosswalk
- `left` -> 0:left
- `leftdw` -> 0:left
- `obs` -> 5:obstacle
- `obs1` -> 5:obstacle
- `obs2` -> 5:obstacle
- `obstacle` -> 5:obstacle
- `park` -> 3:park
- `person` -> 4:person
- `return` -> 2:turnaround
- `right` -> 1:right
- `sideway` -> 6:crosswalk
- `stop` -> skip
- `turnaround` -> 2:turnaround

YOLO source-specific remapping overrides:

- `g01_yolo_09` `obs1` -> 3:park
- `g01_yolo_10` `obs1` -> 3:park
- `g01_yolo_14` `obs1` -> 3:park

## LaneNet Summary

- accepted images: 13241
- split counts: test:1325, train:10592, val:1324
- mask generation: LabelMe line/linestrip annotations rasterized with width 5px; polygon annotations filled
- total lane shape instances: 27147

| source | seen labels | accepted images | skipped | shape labels |
| --- | ---: | ---: | --- | --- |
| g01_lanenet | 2971 | 2971 | - | line:1:2856, line:2:2792 |
| g04_lanenet | 5000 | 4989 | duplicate_image:6, no_lane_shapes:5 | line:line:7374, linestrip:line:1719, polygon:1:2, polygon:line:1841 |
| g23_roadline | 1690 | 1689 | duplicate_image:1 | line:d:1, line:roadline:3106, line:turen:41 |
| liu_images01 | 1925 | 1903 | no_lane_shapes:22 | line:line:4267 |
| liu_images02 | 1690 | 1689 | duplicate_image:1 | line:d:1, line:line:3106, line:turen:41 |

## Notes

- Skipped group 23 YOLO training copies under 4.模型训练/yolo to avoid duplicate sign data.
- Skipped 16组 车道线数据标注结果1(1).zip because it duplicates 结果1.zip.
- Imported group 04 YOLO labels from Ultralytics .cache files because raw .txt labels were not included.
- Skipped .7z archives; no 7z reader is required for the usable labeled data found.
- Applied corrected park relabel sets from data/relabel/park_yolo_corrected, data/relabel/park_yolo_g01_yolo_10_corrected: matched 471, kept 445, removed 26 empty samples.
- Stop sign labels were left out because the current project YOLO config has no `stop` class.
- If the task later adds a stop-sign behavior, add a class to the config and rerun this script with an updated mapping.
- Generated datasets under data/yolo and data/lanenet_hnet are ignored by git; keep the script and this report as the reproducible source of truth.
