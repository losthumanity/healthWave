-- Grant all privileges to the user on the database
GRANT ALL PRIVILEGES ON healthwave_db.* TO 'losthumanity'@'%';
FLUSH PRIVILEGES;

-- Create a simple test table to verify connection
USE healthwave_db;
CREATE TABLE IF NOT EXISTS connection_test (
    id INT AUTO_INCREMENT PRIMARY KEY,
    test_message VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert a test record
INSERT INTO connection_test (test_message) VALUES ('Database connection successful');