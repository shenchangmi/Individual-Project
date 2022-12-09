class conv(nn.Module):
    def __init__(
        self,
        state_shape: Union[int, Sequence[int]],
        action_shape: Union[int, Sequence[int]] = 0,
        hidden_sizes: Sequence[int] = (),
        hidden_sizes_conv: Sequence[int] = (),
        norm_layer: Optional[ModuleType] = None,
        activation: Optional[ModuleType] = nn.ReLU,
        device: Union[str, int, torch.device] = "cpu",
        softmax: bool = False,
        concat: bool = False,
        num_atoms: int = 1,
        dueling_param: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
        linear_layer: Type[nn.Linear] = nn.Linear,
    ) -> None:
        super().__init__()
        self.device = device
        self.softmax = softmax
        self.num_atoms = num_atoms
        # in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1

        self.model_conv = nn.Sequential(
            nn.Conv2d(state_shape[0],hidden_sizes_conv[0],5),
            nn.ReLU(),
            nn.Conv2d(hidden_sizes_conv[0],hidden_sizes_conv[1],5),
            nn.ReLU()
        )
        
    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        """Mapping: obs -> flatten (inside MLP)-> logits."""
        
        if self.device is not None:
            obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)

        obs2mlp = self.model_conv(obs)

        return obs2mlp

class Net_conv(nn.Module):

    def __init__(
        self,
        state_shape: Union[int, Sequence[int]],
        action_shape: Union[int, Sequence[int]] = 0,
        hidden_sizes: Sequence[int] = (),
        hidden_sizes_conv: Sequence[int] = (),
        norm_layer: Optional[ModuleType] = None,
        activation: Optional[ModuleType] = nn.ReLU,
        device: Union[str, int, torch.device] = "cpu",
        softmax: bool = False,
        concat: bool = False,
        num_atoms: int = 1,
        dueling_param: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
        linear_layer: Type[nn.Linear] = nn.Linear,
    ) -> None:
        super().__init__()
        self.device = device
        self.softmax = softmax
        self.num_atoms = num_atoms
        # in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1

        self.model_conv = nn.Sequential(
            nn.Conv2d(state_shape[0],hidden_sizes_conv[0],3),
            nn.ReLU(),
            nn.Conv2d(hidden_sizes_conv[0],hidden_sizes_conv[1],3),
            nn.ReLU()
        )

        
        # self.model_rnn = nn.LSTM(state_shape[1], state_shape[1]*2)
        
        obs = np.array([np.zeros(state_shape)])
        if self.device is not None:
            obs = torch.as_tensor(obs, dtype=torch.float32)

        output_dim_conv = self.model_conv(obs).shape
        # output_dim_conv=(hidden_sizes_conv[1],11,11)
        input_dim = int(np.prod(output_dim_conv))
        action_dim = int(np.prod(action_shape)) * num_atoms
        if concat:
            input_dim += action_dim
        self.use_dueling = dueling_param is not None
        output_dim = action_dim if not self.use_dueling and not concat else 0

        self.model = MLP(
            input_dim, output_dim, hidden_sizes, norm_layer, activation, device,
            linear_layer
        )
        self.output_dim = self.model.output_dim
        if self.use_dueling:  # dueling DQN
            q_kwargs, v_kwargs = dueling_param  # type: ignore
            q_output_dim, v_output_dim = 0, 0
            if not concat:
                q_output_dim, v_output_dim = action_dim, num_atoms
            q_kwargs: Dict[str, Any] = {
                **q_kwargs, "input_dim": self.output_dim,
                "output_dim": q_output_dim,
                "device": self.device
            }
            v_kwargs: Dict[str, Any] = {
                **v_kwargs, "input_dim": self.output_dim,
                "output_dim": v_output_dim,
                "device": self.device
            }
            self.Q, self.V = MLP(**q_kwargs), MLP(**v_kwargs)
            self.output_dim = self.Q.output_dim

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        """Mapping: obs -> flatten (inside MLP)-> logits."""
        
        if self.device is not None:
            obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)

        obs2mlp = self.model_conv(obs)
        obs2mlp.flatten()
        logits = self.model(obs2mlp)
        bsz = logits.shape[0]

        if self.num_atoms > 1:
            logits = logits.view(bsz, -1, self.num_atoms) # reshape
        if self.softmax:
            logits = torch.softmax(logits, dim=-1)
        return logits, state

class Net_rnn(nn.Module):

    def __init__(
        self,
        state_shape: Union[int, Sequence[int]],
        action_shape: Union[int, Sequence[int]] = 0,
        hidden_sizes: Sequence[int] = (),
        hidden_sizes_conv: Sequence[int] = (),
        norm_layer: Optional[ModuleType] = None,
        activation: Optional[ModuleType] = nn.ReLU,
        device: Union[str, int, torch.device] = "cpu",
        softmax: bool = False,
        concat: bool = False,
        num_atoms: int = 1,
        dueling_param: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
        linear_layer: Type[nn.Linear] = nn.Linear,
    ) -> None:
        super().__init__()
        self.device = device
        self.softmax = softmax
        self.num_atoms = num_atoms
        # in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1

        self.rnn_hidden_size = state_shape[1]
        self.batch_size = 1

        self.model_rnn = nn.LSTM(state_shape[1], self.rnn_hidden_size)
        
        obs = np.array([np.zeros(state_shape)])
        if self.device is not None:
            obs = torch.as_tensor(obs, dtype=torch.float32)

        input_dim = int(np.prod(self.rnn_hidden_size))
        action_dim = int(np.prod(action_shape)) * num_atoms
        if concat:
            input_dim += action_dim
        self.use_dueling = dueling_param is not None
        output_dim = action_dim if not self.use_dueling and not concat else 0

        self.model = MLP(
            input_dim, output_dim, hidden_sizes, norm_layer, activation, device,
            linear_layer
        )
        self.output_dim = self.model.output_dim

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        """Mapping: obs -> flatten (inside MLP)-> logits."""
        
        if self.device is not None:
            obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
            
        batch_size = obs.shape[0]
        length = obs.shape[1]
        input_size = obs.shape[2]
        # 2个LSTM层，batch_size=3, 隐藏层的特征维度20   
        # h0 = torch.zeros(1, self.batch_size,  self.rnn_hidden_size).to(self.device) 
        # c0 = torch.zeros(1, self.batch_size,  self.rnn_hidden_size).to(self.device)
        h0 = torch.zeros(1, batch_size,  self.rnn_hidden_size).to(self.device) 
        c0 = torch.zeros(1, batch_size,  self.rnn_hidden_size).to(self.device)

        obs_t = torch.zeros(length,batch_size,input_size).to(self.device)

        for i in range(batch_size):
            obs_t[:,i,:] = obs[i,:,:]
        
        # Forward propagate LSTM
        # out, _ = self.lstm(obs, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        output, (h_n, c_n) = self.model_rnn(obs_t, (h0, c0))
        obs2mlp = h_n[0]
        obs2mlp.flatten()
        logits = self.model(obs2mlp)
        bsz = logits.shape[0]

        if self.num_atoms > 1:
            logits = logits.view(bsz, -1, self.num_atoms) # reshape
        if self.softmax:
            logits = torch.softmax(logits, dim=-1)
        return logits, state

