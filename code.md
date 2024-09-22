code:

- llama_backend_init()
	- ggml_time_init()
	- ggml_context* ctx = ggml_init(param)
	- ggml_free(ctx)
- llama_numa_init(params.numa)
- llama_init_from_gpt_params(params); // ==得到ctx和model==
- model = llama_init.model; // llama_model * model
- ctx = llama_init.context; // llama_context * ctx
- 设置cpu线程
	- struct ggml_threadpool_params tpp_batch = ggml_threadpool_params_from_cpu_params(params.cpuparams_batch);
	- struct ggml_threadpool_params tpp = ggml_threadpool_params_from_cpu_params(params.cpuparams);
- 设置优先级
	- set_process_priority(params.cpuparams.priority);
- llama_attach_threadpool(ctx, threadpool, threadpool_batch);
- 如果设置了path_prompt_cache
	- llama_state_load_file()
- add_bos = llama_add_bos_token(model);
- ==std::vector<llama_token> embd_inp;==
- embd_inp = ::llama_tokenize(ctx, prompt, true, true);
- sampler
	- auto & sparams = params.sparams;
	- smpl = gpt_sampler_init(model, sparams);
- std::vector<llama_token> embd;
- while
	- 如果embd非空，则推理 // --- **首次为空， 不会进此循环**
		- ==有超出上下文的策略==
		- params.n_batch：每次推理的最大token数，可能分批往下送
			- n_eval 为 当次送下去的长度
			- llama_decode(ctx, llama_batch_get_one(&embd[i], n_eval, n_past, 0))
			- n_past += n_eval; ==n_past为推理过的长度==
	- embd.clear()
	- 采样
		- const llama_token id = gpt_sampler_sample(smpl, ctx, -1);
		- gpt_sampler_accept(smpl, id, /* accept_grammar= */ true);
		- embd.push_back(id);
		- ==input_echo = true;==
		- --n_remain;
	- 显示
		- for (auto id : embd)
			- const std::string token_str = llama_token_to_piece(ctx, id, params.special);
			- LOG("%s", token_str.c_str());
			- output_tokens.push_back(id);
			- output_ss << token_str; ==???==
	- ==embd_inp, 和 n_consumed==
		- n_consumed初始值为0， 记录是从embd_inp中放入embd的数量累计
		- embd_inp初始时，已经放入了系统提示词 （==system, user, assistant==）
		- 将embd_inp中的所有token放入embd中，更新n_consumed。n_consumed应当为embd_inp.size
	- if (==n_past > 0== && is_interacting) 时，才用户输入 // ==**所以第一次实际推理是将system提示词走了一遍**==
		- 将用户输入，在前后拼接user/assitant等format信息
		- tokenize为input token
		- n_remain -= input_token.size()  // ==n_remain初始值为-1==
		- 


# is_interacting是变化的bool，是最扯淡的。


# llama_decode
- llama_decode(ctx, batch)
	- llama_decode_internal(lctx, batch);
		- lctx.embd_seq.clear();
		- llama_output_reserve(lctx, n_outputs) 
			- lctx.buf_output = ggml_backend_buft_alloc_buffer(llama_default_buffer_type_cpu(true), new_size);
			- float * output_base = (float *) ggml_backend_buffer_get_base(lctx.buf_output);
			- ==lctx.logits== = has_logits ? output_base : nullptr;
			- lctx.logits_size = logits_size;
		- ggml_cgraph * gf = llama_build_graph(lctx, ubatch, false); // ==无必要，每次都重新build吧，可思考==
		- ==struct ggml_tensor * res = ggml_graph_node(gf, -1);==
			- cgraph->nodes\[i]
		- struct ggml_tensor * embd = ggml_graph_node(gf, -2);
		- ggml_backend_sched_alloc_graph(lctx.sched, gf);
		- llama_set_inputs(lctx, ubatch);
		- llama_graph_compute(lctx, gf, n_threads, threadpool);
		- if (res) {
			- ggml_backend_t **backend_res** = ggml_backend_sched_get_tensor_backend(lctx.sched, res);
			- float * logits_out = ==lctx.logits== + n_outputs_prev*n_vocab;
			- const int32_t n_outputs_new = lctx.n_outputs;
			- if (n_outputs_new) {
				- ggml_backend_tensor_get_async(backend_res, res, logits_out, 0, n_outputs_new*n_vocab*sizeof(float));
					- ggml_backend_tensor_get（tensor, data, offset, size）
						- buf->iface.get_tensor(buf, tensor, data, offset, size);
							- ggml_backend_cpu_buffer_get_tensor
								- memcpy(data, (const char *)tensor->data + offset, size);
		- n_outputs_prev += lctx.n_outputs;
		- lctx.n_outputs = n_outputs;
		- 

## 关于拿到tensor数据的地方



# 结果采样

- const auto * logits = llama_get_logits_ith(ctx, -1);
	- llama_synchronize(ctx);
		- ggml_backend_sched_synchronize(ctx->sched);
			- for (int i = 0; i < sched->n_backends; i++) {
				- ggml_backend_synchronize(sched->backends[i]);
					- backend->iface.synchronize(backend);
	- ctx->logits
- x
