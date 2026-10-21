
import torch
from attention import Attention


def main():
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    attention = Attention(
        embedding_dim=16,
        num_heads=4,
        max_seq_len=16
    ).to(device)

    attention.eval()

    x = torch.randn(2, 5, 16, device=device)

    attention.setup_cache(
        batch_size=2,
        dtype=x.dtype
    )

    with torch.inference_mode():
        # Test 1: KV Cacheなしで5Tokenを処理
        full_output = attention(x)

        # Test 2: 最初の4Tokenを処理（Prefill）
        prefill_output = attention(
            x[:, :4, :],
            use_cache=True
        )

        cache = attention.kv_cache
        assert cache is not None
        assert cache.size == 4

        # Test 3: 最後の1Tokenを処理（Decode）
        decode_output = attention(
            x[:, 4:5, :],
            use_cache=True
        )

        assert cache.size == 5

        # PrefillとDecodeの結果を結合
        cached_output = torch.cat(
            [prefill_output, decode_output],
            dim=1
        )

        # Test 4: 通常実行とキャッシュ利用の結果を比較
        max_diff = (full_output - cached_output).abs().max().item()

        print("Device:", device)
        print("Full shape:", full_output.shape)
        print("Cached shape:", cached_output.shape)
        print("Max difference:", max_diff)

        assert torch.allclose(
            full_output,
            cached_output,
            atol=1e-5,
            rtol=1e-5
        ), "KV Cache output does not match full attention."

        print("[PASS] KV Cache output matches!")

        # Test 5: リセット
        attention.reset_cache()

        assert cache.size == 0

        print("[PASS] KV Cache reset!")
        print("All tests passed!")


if __name__ == "__main__":
    main()
