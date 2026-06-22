// JavaDemo.java
package com.anvay.eval;

public class OrderService {
    private final String dbUrl;

    public OrderService(String dbUrl) {
        this.dbUrl = dbUrl;
    }

    public boolean placeOrder(OrderRequest req) {
        if (req == null || req.amount() <= 0) {
            return false;
        }
        return true;
    }
}

record OrderRequest(String itemId, double amount) {}
