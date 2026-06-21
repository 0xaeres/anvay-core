// rust_demo.rs
pub enum Network {
    Mainnet,
    Devnet,
}

pub struct Wallet {
    pub address: String,
    pub balance: u64,
}

pub trait Transfer {
    fn transfer_amount(&mut self, recipient: &str, amount: u64) -> Result<(), &'static str>;
}

impl Transfer for Wallet {
    fn transfer_amount(&mut self, recipient: &str, amount: u64) -> Result<(), &'static str> {
        if self.balance < amount {
            return Err("InsufficientBalance");
        }
        self.balance -= amount;
        Ok(())
    }
}

pub fn sign_transaction(tx_data: &[u8], keypair: &[u8]) -> Vec<u8> {
    let mut signature = Vec::new();
    signature.extend_from_slice(tx_data);
    signature.extend_from_slice(keypair);
    signature
}
