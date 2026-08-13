# 8月13日

运行代码 **注意两个数据文件要分开两个跑的**
```
python run_blind_annotator.py \
  --tasks de_third_model_tasks.jsonl \
  --image-root  '下载图片的位置' \
  --out-dir stage3_runs/de/model_c_gpt5.6 \
  --annotator-id model_c_gpt5.6 \
  --host '你的base url' \
  --api-key '你的api_key' \
  --model gpt-5.6-sol \
  --question-language en \
  --option-language en \
  --max-questions-per-batch 1 \
  --concurrency 2 \
  --skip-done
```

```
python run_blind_annotator.py \
  --tasks de_third_model_tasks.jsonl \
  --image-root  '下载图片的位置' \
  --out-dir stage3_runs/au/model_c_gpt5.6 \
  --annotator-id model_c_gpt5.6 \
  --host '你的base url' \
  --api-key '你的api_key' \
  --model gpt-5.6-sol \
  --question-language en \
  --option-language en \
  --max-questions-per-batch 1 \
  --concurrency 2 \
  --skip-done
```

1. **out-dir可以指定一下或者就默认也行，最后把stage3_runs打包发给我**
2. **如果出错了就直接重跑同一个指令就行，自动跳过done的内容**

# 8月5日

1. 下载图片：`https://huggingface.co/datasets/p1k0/HBG`里面的`stage3_image_supplement_110.tar`，`hbg_stage3_images.tar.zst`
2. 运行代码 **注意两个数据文件要分开两个跑的**
```
python run_blind_annotator.py \
  --tasks de_blind_tasks.jsonl \
  --image-root  '下载图片的位置' \
  --out-dir stage3_runs/de/model_b_qwen3.8 \
  --annotator-id model_b_qwen3.8 \
  --host '你的base url' \
  --api-key '你的api_key' \
  --model qwen3.8-max \
  --question-language en \
  --option-language en \
  --max-questions-per-batch 1 \
  --concurrency 4 \
  --skip-done
```

```
python run_blind_annotator.py \
  --tasks au_blind_tasks.jsonl \
  --image-root  '下载图片的位置' \
  --out-dir stage3_runs/au/model_b_qwen3.8 \
  --annotator-id model_b_qwen3.8 \
  --host '你的base url' \
  --api-key '你的api_key' \
  --model qwen3.8-max \
  --question-language en \
  --option-language en \
  --max-questions-per-batch 1 \
  --concurrency 4 \
  --skip-done
```

1. **out-dir可以指定一下或者就默认也行，最后把stage3_runs打包发给我**
2. **如果出错了就直接重跑同一个指令就行，自动跳过done的内容**
