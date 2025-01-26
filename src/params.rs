use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::tensor::{TensorInfo, TensorView};
use safetensors::{Dtype, SafeTensors};
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    // pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
    //     // todo!("实现从safetensors文件的模型参数加载");
    //     let get_tensor = |name: &str| -> Tensor<f32> {
    //         let tensor_view = safetensor
    //             .tensor(name)
    //             .expect(&format!("Tensor {} not found", name));
    //         let data: Vec<f32> = match tensor_view.dtype() {
    //             safetensors::Dtype::F32 => {
    //                 let mut data = Vec::new();
    //                 let element_size = std::mem::size_of::<f32>();
    //                 for i in (0..tensor_view.data().len()).step_by(element_size) {
    //                     let slice = &tensor_view.data()[i..i + element_size];
    //                     let value = f32::from_le_bytes(slice.try_into().unwrap());
    //                     data.push(value);
    //                 }
    //                 data
    //             }
    //             _ => panic!("Unsupported data type"),
    //         };
    //         let shape: Vec<usize> = tensor_view.shape().to_vec();
            // Tensor::new(data, &shape)
    //     };

    //     let num_hidden_layers = config.num_hidden_layers;
    //     let hidden_size = config.hidden_size;
    //     let intermediate_size = config.intermediate_size;
    //     let vocab_size = config.vocab_size;
    //     let num_attention_heads = config.num_attention_heads;
    //     let num_key_value_heads = config.num_key_value_heads;

    //     let head_size = hidden_size / num_attention_heads;
    //     let kv_head_size = hidden_size / num_key_value_heads;

    //     LLamaParams {
    //         embedding_table: get_tensor("lm_head.weight"), // 修改为 lm_head.weight
    //         rms_att_w: (0..num_hidden_layers)
    //             .map(|i| {
    //                 get_tensor(&format!(
    //                     "model.layers.{}.post_attention_layernorm.weight",
    //                     i
    //                 ))
    //             })
    //             .collect(),
    //         wq: (0..num_hidden_layers)
    //             .map(|i| get_tensor(&format!("model.layers.{}.self_attn.q_proj.weight", i)))
    //             .collect(),
    //         wk: (0..num_hidden_layers)
    //             .map(|i| get_tensor(&format!("model.layers.{}.self_attn.k_proj.weight", i)))
    //             .collect(),
    //         wv: (0..num_hidden_layers)
    //             .map(|i| get_tensor(&format!("model.layers.{}.self_attn.v_proj.weight", i)))
    //             .collect(),
    //         wo: (0..num_hidden_layers)
    //             .map(|i| get_tensor(&format!("model.layers.{}.self_attn.o_proj.weight", i)))
    //             .collect(),
    //         rms_ffn_w: (0..num_hidden_layers)
    //             .map(|i| get_tensor(&format!("model.layers.{}.input_layernorm.weight", i)))
    //             .collect(),
    //         w_up: (0..num_hidden_layers)
    //             .map(|i| get_tensor(&format!("model.layers.{}.mlp.up_proj.weight", i)))
    //             .collect(),
    //         w_gate: (0..num_hidden_layers)
    //             .map(|i| get_tensor(&format!("model.layers.{}.mlp.gate_proj.weight", i)))
    //             .collect(),
    //         w_down: (0..num_hidden_layers)
    //             .map(|i| get_tensor(&format!("model.layers.{}.mlp.down_proj.weight", i)))
    //             .collect(),
    //         rms_out_w: get_tensor("model.norm.weight"), // 修改为 model.norm.weight
    //         lm_head: get_tensor("lm_head.weight"),
    //     }
    // }
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        // 辅助函数封装
        fn convert_tensor(safetensor: &SafeTensors, name: &str) -> Tensor<f32> {
            let view = safetensor.tensor(name)
                .unwrap_or_else(|_| panic!("Missing required tensor: {}", name));
            
            assert_eq!(view.dtype(), Dtype::F32, "Tensor {} is not F32 type", name);
            
            // 字节到f32转换
            let data = view.data();
            let mut f32_data = Vec::with_capacity(data.len() / 4);
            for chunk in data.chunks_exact(4) {
                f32_data.push(f32::from_le_bytes(chunk.try_into().unwrap()));
            }
            
            Tensor::new(f32_data, view.shape())
        }

        // 调试打印所有key（实际使用时可以注释掉）
        println!("===== Available Tensors =====");
        for key in safetensor.names() {
            println!("key: {}", key);
        }
        println!("=============================");

        // 处理共享词嵌入情况：使用lm_head.weight作为输入词表
        let embedding_table = convert_tensor(safetensor, "lm_head.weight");
        
        // 验证词表维度
        assert_eq!(
            embedding_table.shape()[0],
            config.vocab_size as usize,
            "lm_head.weight形状{:?}与config.vocab_size {}不匹配",
            embedding_table.shape(),
            config.vocab_size
        );

        // 初始化各层参数容器
        let num_layers = config.num_hidden_layers as usize;
        let mut rms_att_w = Vec::with_capacity(num_layers);
        let mut wq = Vec::with_capacity(num_layers);
        let mut wk = Vec::with_capacity(num_layers);
        let mut wv = Vec::with_capacity(num_layers);
        let mut wo = Vec::with_capacity(num_layers);
        let mut rms_ffn_w = Vec::with_capacity(num_layers);
        let mut w_up = Vec::with_capacity(num_layers);
        let mut w_gate = Vec::with_capacity(num_layers);
        let mut w_down = Vec::with_capacity(num_layers);

        // 逐层加载参数
        for layer_idx in 0..num_layers {
            let layer_prefix = format!("model.layers.{}", layer_idx);

            // 自注意力层参数
            rms_att_w.push(convert_tensor(
                safetensor,
                &format!("{}.input_layernorm.weight", layer_prefix)
            ));

            wq.push(convert_tensor(
                safetensor,
                &format!("{}.self_attn.q_proj.weight", layer_prefix)
            ));

            wk.push(convert_tensor(
                safetensor,
                &format!("{}.self_attn.k_proj.weight", layer_prefix)
            ));

            wv.push(convert_tensor(
                safetensor,
                &format!("{}.self_attn.v_proj.weight", layer_prefix)
            ));

            wo.push(convert_tensor(
                safetensor,
                &format!("{}.self_attn.o_proj.weight", layer_prefix)
            ));

            // FFN层参数
            rms_ffn_w.push(convert_tensor(
                safetensor,
                &format!("{}.post_attention_layernorm.weight", layer_prefix)
            ));

            w_up.push(convert_tensor(
                safetensor,
                &format!("{}.mlp.up_proj.weight", layer_prefix)
            ));

            w_gate.push(convert_tensor(
                safetensor,
                &format!("{}.mlp.gate_proj.weight", layer_prefix)
            ));

            w_down.push(convert_tensor(
                safetensor,
                &format!("{}.mlp.down_proj.weight", layer_prefix)
            ));
        }

        // 输出层参数
        let rms_out_w = convert_tensor(safetensor, "model.norm.weight");
        
        // 最终参数结构
        LLamaParams {
            embedding_table,  // 使用lm_head.weight作为输入嵌入
            rms_att_w,
            wq,
            wk,
            wv,
            wo,
            rms_ffn_w,
            w_up,
            w_gate,
            w_down,
            rms_out_w,
            lm_head: convert_tensor(safetensor, "lm_head.weight"), // 共享权重
        }
    }
}
