import torch

from helper_lib.checkpoints import save_checkpoint
from helper_lib.rl_lm import Environment, Policy, SimpleNet, build_vocab

if __name__ == "__main__":
    # This workload is many tiny sequential ops (one word at a time), not big
    # batched matmuls -- torch's default multi-threading adds pure overhead
    # here (~24x slower measured on this machine), so pin to one thread.
    torch.set_num_threads(1)

    VOCAB_SIZE = 1000
    CONTEXT_SIZE = 3
    HIDDEN_DIM = 256
    MAX_SEQ_LENGTH = 5
    START_WORD = "group"
    END_LETTER = "e"

    EPOCHS = 1000
    BATCH_SIZE = 30
    LR = 1e-2

    vocab, word_to_idx, idx_to_word = build_vocab(vocab_size=VOCAB_SIZE)

    env = Environment(word_to_idx, idx_to_word, max_seq_length=MAX_SEQ_LENGTH, end_letter=END_LETTER, context_size=CONTEXT_SIZE)

    model = SimpleNet(input_dim=CONTEXT_SIZE * VOCAB_SIZE, hidden_dim=HIDDEN_DIM, action_dim=VOCAB_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    policy = Policy(model, optimizer, env, vocab, idx_to_word, start_word=START_WORD, batch_size=BATCH_SIZE)

    reward_history = []
    for epoch in range(EPOCHS):
        batch_loss, batch_rets, batch_lens = policy.train_one_epoch()
        avg_ret = sum(batch_rets) / len(batch_rets)
        reward_history.append(avg_ret)
        if epoch % 100 == 0:
            print(f"epoch: {epoch:4d} \t loss: {batch_loss:.3f} \t return: {avg_ret:.3f} \t ep_len: {sum(batch_lens) / len(batch_lens):.3f}")

    save_checkpoint(model, optimizer, EPOCHS, batch_loss.item(), avg_ret, checkpoint_dir="checkpoints_rl/best", filename="policy.pth")

    print("Finished Training")
    print("Sample chains from the trained policy:")
    for _ in range(5):
        print(" -> ".join(policy.generate_chain()))
