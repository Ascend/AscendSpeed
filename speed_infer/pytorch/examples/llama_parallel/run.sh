SCRIPT_DIR=$(cd $(dirname $0); pwd)
MODEL_TARGET_DIR=$SCRIPT_DIR
# SCRIPT_PATH=$SCRIPT_DIR/transformers_patch/layer/modeling_llama_layer_performance.py
TRANSFORMER_PACKAGE_PATH=$(python3 -c 'import transformers; import os; print(os.path.dirname(transformers.__file__))')
RUN_OPTION_LIST="--run --performance --webdemo --zhipu --profiling"
MODEL_LIST="--llama1-13b_parallel"

function fn_prepare_llama1_13b_parallel()
{
    if [ ! -f "$MODEL_TARGET_DIR/tokenizer/tokenizer.model" ];then
        mkdir $MODEL_TARGET_DIR/part_model
        mkdir $MODEL_TARGET_DIR/part_model/0
        mkdir $MODEL_TARGET_DIR/part_model/1
        mkdir $MODEL_TARGET_DIR/tokenizer
        # part_model/0/
        ln -s /data/acltransformer_testdata/weights/llama/llama-13b-part_model_2/part_model/0/config.json $MODEL_TARGET_DIR/part_model/0/config.json
        ln -s /data/acltransformer_testdata/weights/llama/llama-13b-part_model_2/part_model/0/generation_config.json $MODEL_TARGET_DIR/part_model/0/generation_config.json
        ln -s /data/acltransformer_testdata/weights/llama/llama-13b-part_model_2/part_model/0/pytorch_model.bin.index.json $MODEL_TARGET_DIR/part_model/0/pytorch_model.bin.index.json
        ln -s /data/acltransformer_testdata/weights/llama/llama-13b-part_model_2/part_model/0/pytorch_model-00001-of-00003.bin $MODEL_TARGET_DIR/part_model/0/pytorch_model-00001-of-00003.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama-13b-part_model_2/part_model/0/pytorch_model-00002-of-00003.bin $MODEL_TARGET_DIR/part_model/0/pytorch_model-00002-of-00003.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama-13b-part_model_2/part_model/0/pytorch_model-00003-of-00003.bin $MODEL_TARGET_DIR/part_model/0/pytorch_model-00003-of-00003.bin
        # part_model/1/
        ln -s /data/acltransformer_testdata/weights/llama/llama-13b-part_model_2/part_model/1/config.json $MODEL_TARGET_DIR/part_model/1/config.json
        ln -s /data/acltransformer_testdata/weights/llama/llama-13b-part_model_2/part_model/1/generation_config.json $MODEL_TARGET_DIR/part_model/1/generation_config.json
        ln -s /data/acltransformer_testdata/weights/llama/llama-13b-part_model_2/part_model/1/pytorch_model.bin.index.json $MODEL_TARGET_DIR/part_model/1/pytorch_model.bin.index.json
        ln -s /data/acltransformer_testdata/weights/llama/llama-13b-part_model_2/part_model/1/pytorch_model-00001-of-00003.bin $MODEL_TARGET_DIR/part_model/1/pytorch_model-00001-of-00003.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama-13b-part_model_2/part_model/1/pytorch_model-00002-of-00003.bin $MODEL_TARGET_DIR/part_model/1/pytorch_model-00002-of-00003.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama-13b-part_model_2/part_model/1/pytorch_model-00003-of-00003.bin $MODEL_TARGET_DIR/part_model/1/pytorch_model-00003-of-00003.bin
        # tokenizer/
        ln -s /data/acltransformer_testdata/weights/llama/llama-13b-part_model_2/tokenizer/special_tokens_map.json $MODEL_TARGET_DIR/tokenizer/special_tokens_map.json
        ln -s /data/acltransformer_testdata/weights/llama/llama-13b-part_model_2/tokenizer/tokenizer_config.json $MODEL_TARGET_DIR/tokenizer/tokenizer_config.json
        ln -s /data/acltransformer_testdata/weights/llama/llama-13b-part_model_2/tokenizer/tokenizer.model $MODEL_TARGET_DIR/tokenizer/tokenizer.model
    fi

    cp $MODEL_TARGET_DIR/modeling_llama_parallel_performance.py $TRANSFORMER_PACKAGE_PATH/models/llama/modeling_llama.py
}

function fn_clean()
{
    rm $MODEL_TARGET_DIR/pytorch_model*
    rm $MODEL_TARGET_DIR/*.json
    rm $MODEL_TARGET_DIR/*.model
}

function fn_main()
{
    echo "-----run.sh-----"
    fn_clean
    
    if [[ ! -z "$1" ]];then
        RUN_OPTION=$1
        echo "[RUN_OPTION]: $RUN_OPTION"
        shift
    fi
    
    if [[ ! -z "$1" ]];then
        MODEL=$1
        echo "[MODEL]: $MODEL"
        shift
    fi
    
    # if [[ ! -z "$1" ]];then
    #     TEMP_SCRIPT_PATH="$1"
    #     if [[ ! -e $TEMP_SCRIPT_PATH ]];then
    #         SCRIPT_PATH=$TEMP_SCRIPT_PATH
    #         echo "[MODEL_SCRIPT_PATH]: $SCRIPT_PATH"
    #         shift
    #     else
    #         echo "WRONG dir"
    #         exit -1
    #     fi
    # fi

    cd $SCRIPT_DIR
    # echo "[TRANSFORMER_PACKAGE_PATH]: $TRANSFORMER_PACKAGE_PATH"

    case "${MODEL}" in
        "--llama1-13b_parallel")
            fn_prepare_llama1_13b_parallel
            ;;
        "--help")
            echo "run.sh [--run|--performance|--webdemo|--zhipu|--profiling] [--llama1-13b_parallel] [model script path]"
            ;;
        *)
            echo "unknown build type:${MODEL}"
            echo "run.sh [--run|--performance|--webdemo|--zhipu|--profiling] [--llama1-13b_parallel] [model script path]"
            exit -1
            ;;
    esac

    # cp $SCRIPT_PATH $TRANSFORMER_PACKAGE_PATH/models/llama/modeling_llama.py

    case "${RUN_OPTION}" in
        "--run")
            ;;
        "--performance")
            ;;
        "--webdemo")
            ;;
        "--zhipu")
            case "${MODEL}" in
                "--llama1-13b_parallel")
                    echo "start llama-13b-parallel"
                    torchrun --nproc_per_node 2 $SCRIPT_DIR/zhipu_test.py
                    ;;
                "--help")
                    echo "run.sh [--run|--performance|--webdemo|--zhipu|--profiling] [--llama1-13b_parallel] [model script path]"
                    ;;
                *)
                    echo "unknown build type:${MODEL}"
                    echo "run.sh [--run|--performance|--webdemo|--zhipu|--profiling] [--llama1-13b_parallel] [model script path]"
                    exit -1
                    ;;
            esac
            ;;
        "--profiling")
            ;;
        "--precision")
            ;;
        "--help")
            echo "run.sh [--run|--performance|--webdemo|--zhipu|--profiling] [--llama1-13b_parallel] [model script path]"
            ;;
        *)
            echo "unknown build type:${RUN_OPTION}"
            echo "run.sh [--run|--performance|--webdemo|--zhipu|--profiling] [--llama1-13b_parallel] [model script path]"
            exit -1
            ;;
    esac

    fn_clean
}

fn_main "$@"