import torch

from attention import Attention


def main():

    attention = Attention(
        embedding_dim=16,
        num_heads=4,
        max_seq_len=256
    )

    attention.setup_cache(
        batch_size=2,
        dtype=torch.float32
    )

    cache = attention.kv_cache

    assert cache is not None

    print(cache.k_cache.shape)
    print(cache.size)

    attention.reset_cache()

    print(cache.size)


if __name__ == "__main__":
    main()