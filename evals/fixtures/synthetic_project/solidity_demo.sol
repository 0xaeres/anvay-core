// solidity_demo.sol
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IERC20 {
    function transfer(address to, uint256 amount) external returns (bool);
}

library SafeMath {
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        return a + b;
    }
}

contract LiquidityPool {
    using SafeMath for uint256;

    struct PoolInfo {
        uint256 totalLiquidity;
        uint256 feeRate;
    }

    address public owner;
    PoolInfo public pool;

    event TokensSwapped(address indexed user, uint256 amountIn, uint256 amountOut);
    error InsufficientBalance();

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function addLiquidity(uint256 amount) public payable onlyOwner {
        pool.totalLiquidity = pool.totalLiquidity.add(amount);
    }
}
