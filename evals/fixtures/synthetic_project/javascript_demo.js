// javascript_demo.js
class LoggerService {
    constructor(prefix) {
        this.prefix = prefix;
    }

    log(message) {
        console.log(`[${this.prefix}] ${message}`);
    }
}

function formatMessage(msg, level = "INFO") {
    return `${new Date().toISOString()} [${level}] ${msg}`;
}

const defaultLogger = new LoggerService("APP");
