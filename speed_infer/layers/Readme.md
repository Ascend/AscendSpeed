
# Layer接口说明



## MlpGateLayer接口说明

```c++

// 参数说明
struct MlpGateParam {
    int rank = 0; # 必填
    int rankSize = 0; # 必填
    int rankRoot = 0; # 选填
    void *hcclComm = nullptr; # 选填
    atb::infer::ActivationType activationType; # 激活函数
    bool transposeB = false; # 是否weight做转置
    bool isBias = false;  # 是否加bias
    bool isPack = false;  # gate和up 合并计算，内部插split算子
};
```


### MLP 带bias

```c++
// 输入参数顺序
IN_HIDDENSTATES_ID = 0,             // [batch, seqLen, hiddenSize], half
IN_WEIGHT_UP_ID,                    // [hiddenSize, ffnHiddenSize], half
IN_WEIGHT_GATE_ID,                  // [hiddenSize, ffnHiddenSize], half
IN_WEIGHT_DOWN_ID,                  // [ffnHiddenSize, hiddenSize], half
IN_BIAS_UP_ID,                      //
IN_BIAS_GATE_ID,                    //
IN_BIAS_DOWN_ID,                    //
```


### MLP 不带bias

```c++
// 输入参数顺序
IN_HIDDENSTATES_ID = 0,             // [batch, seqLen, hiddenSize], half
IN_WEIGHT_UP_ID,                    // [hiddenSize, ffnHiddenSize], half
IN_WEIGHT_GATE_ID,                  // [hiddenSize, ffnHiddenSize], half
IN_WEIGHT_DOWN_ID,                  // [ffnHiddenSize, hiddenSize], half
```


### MLP 带bias Gate和Up合并

```c++
// 输入参数顺序
IN_HIDDENSTATES_ID = 0,             // [batch, seqLen, hiddenSize], half
IN_WEIGHT_UP_ID,                    // [hiddenSize, ffnHiddenSize], half
IN_WEIGHT_DOWN_ID,                  // [ffnHiddenSize, hiddenSize], half
IN_BIAS_UP_ID,
IN_BIAS_DOWN_ID,
```

### MLP 不带bias Gate和Up合并

```c++
// 输入参数顺序
IN_HIDDENSTATES_ID = 0,             // [batch, seqLen, hiddenSize], half
IN_WEIGHT_UP_ID,                    // [hiddenSize, ffnHiddenSize], half
IN_WEIGHT_DOWN_ID,                  // [ffnHiddenSize, hiddenSize], half
```


## MlpLayer接口说明

```c++
// 参数
struct MlpParam {
    int rank = 0; # 必填
    int rankSize = 0; # 必填
    int rankRoot = 0; # 选填
    void *hcclComm = nullptr; # 选填
    atb::infer::ActivationType activationType; # 激活函数
    bool transposeB = false; # 是否weight做转置
    bool isBias = false;  # 是否加bias
};
```


### MLP 带bias

```c++
// 输入参数顺序
IN_HIDDENSTATES_ID = 0,             // [batch, seqLen, hiddenSize], half
IN_WEIGHT_UP_ID,                    // [hiddenSize, ffnHiddenSize], half
IN_WEIGHT_DOWN_ID,                  // [ffnHiddenSize, hiddenSize], half
IN_BIAS_UP_ID,                      //
IN_BIAS_DOWN_ID,                    //
```


### MLP 不带bias

```c++
// 输入参数顺序
IN_HIDDENSTATES_ID = 0,             // [batch, seqLen, hiddenSize], half
IN_WEIGHT_UP_ID,                    // [hiddenSize, ffnHiddenSize], half
IN_WEIGHT_DOWN_ID,                  // [ffnHiddenSize, hiddenSize], half
```



## RowParallelLinear

```c++
struct ParallelParam {
    int rank = 0;
    int rankSize = 0;
    int rankRoot = 0;
    void *hcclComm = nullptr;
    bool isBias = false;
    bool transposeA = false;
    bool transposeB = false; # weight
};
```

## ColumnParallelLinear

```c++
struct ParallelParam {
    int rank = 0;
    int rankSize = 0;
    int rankRoot = 0;
    void *hcclComm = nullptr;
    bool isBias = false;
    bool transposeA = false;
    bool transposeB = false;
};
```

## ParallelLmHead

```c++
struct ParallelLmHeadParam {
    int rank = 0;
    int rankSize = 1;
    std::string backend = "hccl";
    bool unpadInputs = false;   # unpad场景，无batch维度
    bool gatherAhead = false;   # 提供prefill gather ahead功能，减少lmhead计算量
    bool transposeA = false;
    bool transposeB = false;
};
```


## **FlashAttentionWithROPELayer**



接口暂未支持，待完善

```c++
struct FTWithROPEParam {
    bool isBias = false;
};
```

