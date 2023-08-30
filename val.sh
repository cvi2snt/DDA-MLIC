# Source: AID --> Target: UCM
python main.py --phase test -s AID -t UCM -s-dir <path_to_source_dataset> -t-dir <path_to_target_dataset> --model-path models/

# Source: UCM --> Target: AID
python main.py --phase test -s UCM -t AID -s-dir <path_to_source_dataset> -t-dir <path_to_target_dataset>

# Source: AID --> Target: DFC
python main.py --phase test -s AID -t DFC -s-dir <path_to_source_dataset> -t-dir <path_to_target_dataset>

# Source: UCM --> Target: DFC
python main.py --phase test -s UCM -t DFC -s-dir <path_to_source_dataset> -t-dir <path_to_target_dataset>

# Source: VOC --> Target: Clipart
python main.py --phase test -s VOC -t Clipart -s-dir <path_to_source_dataset> -t-dir <path_to_target_dataset>

# Source: Clipart --> Target: VOC
python main.py --phase test -s Clipart -t VOC -s-dir <path_to_source_dataset> -t-dir <path_to_target_dataset>

# Source: Cityscapes --> Target: Foggycityscapes
python main.py --phase test -s cityscapes -t foggycityscapes -s-dir <path_to_source_dataset> -t-dir <path_to_target_dataset>

