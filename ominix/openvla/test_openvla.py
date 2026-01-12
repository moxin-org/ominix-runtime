import openvla


model = openvla.OpenvlaPipeline(
    dinov2_model_path="/mount/weights/openvla/dinov2.gguf",
    siglip_model_path="/mount/weights/openvla/siglip.gguf",
    proj_model_path="/mount/weights/openvla/proj.gguf",
    llm_model_path="/mount/weights/openvla/llm_q8_0.gguf",
    tokenizer_path="",
    device_name="CUDA0",
    n_threads=4,
    max_nodes=2048,
    ngl=99,
    n_ctx=300
)
print(model.run("/mount/weights/vote_model/2.png", "pick up the black bowl between the plate and the ramekin and place it on the plate"))
print(model.run("/mount/weights/vote_model/2.png", "pick up the black bowl between the plate and the ramekin and place it on the plate"))
print(model.run("/mount/weights/vote_model/2.png", "pick up the black bowl between the plate and the ramekin and place it on the plate"))