from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp


class OTProblemOracle:
    def __init__(self, q: jnp.ndarray, C: jnp.ndarray) -> None:
        """
        :param jnp.ndarray q: Shape (d, ).
        :param jnp.ndarray C: Shape (d, d).
        """
        self.q = q
        self.C = C
        self.C_norm = jnp.max(jnp.abs(self.C))
        self.ones = jnp.ones_like(q)

    def _tree_flatten(self):
        children = ()  # arrays / dynamic values
        aux_data = {"C": self.C, "q": self.q}  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    @jax.jit
    @partial(jax.vmap, in_axes=(0, None, 0, 0))
    def f(self, x: jnp.ndarray, p: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the value of operator at point z = (x, p, u, v).

        :param jnp.ndarray x: Shape (d, d).
        :param jnp.ndarray p: Shape (d, ).
        :param jnp.ndarray u: Shape (d, ).
        :param jnp.ndarray v: Shape (d, ).
        :return jnp.ndarray: float
        """
        return (
            jnp.multiply(self.C + 2 * self.C_norm * (jnp.outer(u, self.ones) + jnp.outer(self.ones, v)), x).sum()
            - jnp.dot(u, p)
            - jnp.dot(v, self.q)
        )

    @jax.jit
    def grad_x(self, x: jnp.ndarray, p: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the value of x gradient at point z = (x, p, u, v).
        """
        return self.C + 2 * self.C_norm * (jnp.outer(u, self.ones) + jnp.outer(self.ones, v))

    @jax.jit
    def grad_p(self, x: jnp.ndarray, p: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the value of p gradient at point z = (x, p, u, v).
        """
        return -u

    @jax.jit
    def grad_u(self, x: jnp.ndarray, p: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the value of u gradient at point z = (x, p, u, v).
        """
        return jnp.matmul(x, self.ones) - p

    @jax.jit
    def grad_v(self, x: jnp.ndarray, p: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the value of v gradient at point z = (x, p, u, v).
        """
        return jnp.matmul(x.T, self.ones) - self.q

    @jax.jit
    def G(self, x: jnp.ndarray, p: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the value of operator (grad_x(z), -grad_y(z)) at point z = (x, p, u, v).
        """
        return jnp.array(
            [self.grad_x(x, p, u, v), self.grad_p(x, p, u, v), -self.grad_u(x, p, u, v), -self.grad_v(x, p, u, v)]
        )


jax.tree_util.register_pytree_node(OTProblemOracle, OTProblemOracle._tree_flatten, OTProblemOracle._tree_unflatten)


# class OperatorOracle:
#     def __init__(self, C: jnp.ndarray, q: jnp.ndarray, n: int):
#         """
#         :param jnp.ndarray C: Shape (T, d, d).
#         :param jnp.ndarray q: Shape (T, d).
#         :param int n: Number of instances to compute.
#         """
#         self.T = C.shape[0]
#         self.n = n
#         assert self.n <= self.T
#         self.C = C
#         self.q = q
#         self.C_norm = jnp.max(jnp.abs(C))

#     def _tree_flatten(self):
#         children = ()  # arrays / dynamic values
#         aux_data = {"C": self.C, "q": self.q, "n": self.n}  # static values
#         return (children, aux_data)

#     @classmethod
#     def _tree_unflatten(cls, aux_data, children):
#         return cls(*children, **aux_data)

#     @jax.jit
#     def f(self, x: jnp.ndarray, p: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
#         p_bar = jnp.stack([p for _ in range(self.T)])
#         return (
#             jnp.multiply(self.C, x)
#             + 2 * self.C_norm * (jnp.multiply(u, x.sum(axis=2) - p_bar) + jnp.multiply(v, x.sum(axis=1) - self.q))
#         ).sum()

#     # def grad_x(self, z: SpacePoint) -> XSpaceType:
#     #     for i in range(self._n):  # tnrange(self._n, desc="Grad x"):
#     #         self._grad_x[i] = self._oracles[i].grad_x(z.x[i], z.p, z.u[i], z.v[i])

#     #     self._grad_x /= self._n
#     #     return self._grad_x

#     # def grad_p(self, z: SpacePoint) -> np.ndarray:
#     #     grad_p = np.zeros_like(self._grad_p)
#     #     for i in range(self._n):  #  tnrange(self._n, desc="Grad p"):
#     #         grad_p += self._oracles[i].grad_p(z.x[i], z.p, z.u[i], z.v[i])

#     #     self._grad_p = grad_p / self._n
#     #     return self._grad_p

#     # def grad_u(self, z: SpacePoint) -> HSpaceType:
#     #     for i in range(self._n):  # tnrange(self._n, desc="Grad u"):
#     #         self._grad_u[i] = self._oracles[i].grad_u(z.x[i], z.p, z.u[i], z.v[i])

#     #     self._grad_u /= self._n
#     #     return self._grad_u

#     # def grad_v(self, z: SpacePoint) -> HSpaceType:
#     #     for i in range(self._n):  # tnrange(self._n, desc="Grad v"):
#     #         self._grad_v[i] = self._oracles[i].grad_v(z.x[i], z.p, z.u[i], z.v[i])

#     #     self._grad_v /= self._n
#     #     return self._grad_v
