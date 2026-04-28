import pickle

import torch


class NonlinearProgram:
    def __init__(self, params_path: str):
        with open(params_path, "rb") as f:
            constraint_parameters = pickle.load(f)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._A: torch.Tensor = (
            torch.stack(constraint_parameters["A"], dim=0)
            .to(torch.float32)
            .to(self.device)
        )
        self._c: torch.Tensor = (
            torch.stack(constraint_parameters["c"], dim=0)
            .to(torch.float32)
            .to(self.device)
        )
        self._d: torch.Tensor = (
            torch.stack(constraint_parameters["d"], dim=0)
            .to(torch.float32)
            .to(self.device)
        )
        self._p: torch.Tensor = (
            constraint_parameters["p"].to(torch.float32).to(self.device)
        )
        self._Q: torch.Tensor = (
            constraint_parameters["Q"].to(torch.float32).to(self.device)
        )

    def eq_constraints(self, x, y):
        """
        Compute the constraints y^T A_i y + c_i^T y + d_i - x_i^2
        for a batch:
        y: (BS, NO)
        x: (BS, NI)
        A: (NC, NO, NO)
        c: (NC, NO)
        d: (NC,)
        returns: (BS, NC)
        """
        # BS, NO = y.shape
        # NC, _, _ = self._A.shape
        # Note, in this problem NC = NI

        # 1) Quadratic term: for each i, y[b]ᵀ A[i] y[b]
        term1 = torch.einsum("bj,ijk,bk->bi", y, self._A, y)
        # 2) Linear term: cᵀ y
        term2 = y @ self._c.T  # [BS,NO] @ [NO,NC] → [BS,NC]
        # 3) constant d (broadcast to [BS,NC])
        term3 = self._d  # shape [NC]
        # 4) xᵢ²
        term4 = x**3  # same shape [BS,NI==NC]

        return term1 + term2 + term3 - term4

    def objective(self, x, y):
        """
        Compute the objective function value.
        0.5 * y^T Q y + p^T sin(y)
        """
        yQ = torch.matmul(y, self._Q)  # shape: (batch_size, dim_y)
        term1 = 0.5 * torch.sum(torch.mul(y, yQ), dim=1)

        # 2) Linear‐sinusoid term: p^T sin(y)
        sin_y = torch.sin(y)
        term2 = torch.sum(torch.mul(self._p, sin_y), dim=1)

        return term1 + term2

    def jacobian(self, y):
        # using einsum to avoid any broadcasting pitfalls:
        #   out[b,i,j] = sum_{k} Msym[i,j,k] * y[b,k]
        # yields (BS,NC,NO)
        return 2 * torch.einsum("ijk,bk->bij", self._A, y) + self._c.unsqueeze(0)


class NonconvexProgram:
    def __init__(self, params_path: str):
        with open(params_path, "rb") as f:
            constraint_parameters = pickle.load(f)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._c: torch.Tensor = (
            constraint_parameters["c"].to(torch.float32).to(self.device)
        )
        self._p: torch.Tensor = (
            constraint_parameters["p"].to(torch.float32).to(self.device)
        )
        self._Q: torch.Tensor = (
            constraint_parameters["Q"].to(torch.float32).to(self.device)
        )

    def eq_constraints(self, x, y):
        """
        Compute the constraints c_i^T y - x_i
        for a batch:
        y: (BS, NO)
        x: (BS, NI)
        c: (NC, NO)
        returns: (BS, NC)
        """
        # BS, NO = y.shape
        # NC, _, _ = self._A.shape
        # Note, in this problem NC = NI

        # 1) Linear term: cᵀ y
        term1 = y @ self._c.T  # [BS,NO] @ [NO,NC] → [BS,NC]

        return term1 - x

    def objective(self, x, y):
        """
        Compute the objective function value.
        0.5 * y^T Q y + p^T sin(y)
        """
        yQ = torch.matmul(y, self._Q)  # shape: (batch_size, dim_y)
        term1 = 0.5 * torch.sum(torch.mul(y, yQ), dim=1)

        # 2) Linear‐sinusoid term: p^T sin(y)
        sin_y = torch.sin(y)
        term2 = torch.sum(torch.mul(self._p, sin_y), dim=1)

        return term1 + term2

    def jacobian(self, y):
        # yields (BS,NC,NO)
        bs = y.shape[0]
        # self._c.shape = (NC, NO)
        # repeat along batch dimension
        return self._c.unsqueeze(0).expand(bs, -1, -1)
