import jax
import jax.numpy as jnp
import numpy as np


def encode(word):
    return np.array([ord(c) - ord('a') for c in word])


def decode(word_vector):
    return ''.join(chr(ord('a') + c) for c in word_vector)

@jax.jit
def score_guess(guess, solution):
    green  = (guess == solution)
    
    # Effectively mask out green numbers
    solution = solution + jnp.where(green, 100, 0)
    guess    = guess + jnp.where(green, 1000, 0)
        
    def is_yellow(i):
        return (jnp.cumsum(guess == guess[i])[i] <= jnp.sum(solution == guess[i]))
    
    yellow = jax.vmap(is_yellow)(jnp.arange(5))
    return jnp.where(green,  2,
           jnp.where(yellow, 1,
                             0))


def fmt_score(score):
    return ''.join('✕~✓'[i] for i in score)

## Test cases
def test(guess, solution, expected_score):
    score = score_guess(encode(guess), encode(solution))
    if np.all(score == np.asarray(expected_score)):
        print(f'Guess {guess} for {solution} scored correctly.')
    else:
        print(f'Guess {guess} for {solution} scored '
              f'{fmt_score(score)}, expected {fmt_score(expected_score)}!')

if __name__ == '__main__':
  test('acccb', 'bccca', [1,2,2,2,1])
  test('aaaab', 'baaaa', [1,2,2,2,1])
  test('abcda', 'abcde', [2,2,2,2,0])
  test('cbada', 'abcde', [1,2,1,2,0])
  test('cbada', 'abcde', [1,2,1,2,0])
  test('zymic', 'could', [0,0,0,0,1])
