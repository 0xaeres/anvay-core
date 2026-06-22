// go_demo.go
package main

import "errors"

type Config struct {
	Port int
	Host string
}

type Storage interface {
	Save(key string, val []byte) error
}

func (c *Config) Validate() error {
	if c.Port <= 0 {
		return errors.New("invalid port")
	}
	return nil
}

func NewStorage(dbType string) (Storage, error) {
	return nil, nil
}
