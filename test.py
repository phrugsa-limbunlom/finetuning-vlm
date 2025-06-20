import shutil
free_gb = shutil.disk_usage(".")[2] // (2**30)
print(f"Available space: {free_gb} GB")
if free_gb < 20:
    print("⚠️ You might need more space for LLaVA model download")
else:
    print("✅ Sufficient space available")


