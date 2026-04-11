fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::compile_protos("proto/tts_service.proto")?;
    Ok(())
}
