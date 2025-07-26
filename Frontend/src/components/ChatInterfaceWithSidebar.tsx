import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Send, MessageCircle, Languages, Menu, PanelLeft, AlertCircle } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import ThemeToggle from "@/components/ThemeToggle";
import { ChatSidebar } from "@/components/ChatSidebar";
import { SidebarProvider, SidebarTrigger, useSidebar } from "@/components/ui/sidebar";
import { useIsMobile } from "@/hooks/use-mobile";
import { useAskQuestion, useHealthCheck } from "@/hooks/use-api";

interface Message {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: Date;
}

interface ChatSession {
  id: string;
  title: string;
  lastMessage: string;
  timestamp: Date;
  messages: Message[];
}

// Custom Sidebar Toggle Component
const CustomSidebarToggle = () => {
  const { toggleSidebar, state } = useSidebar();
  const isMobile = useIsMobile();

  return (
    <Button
      onClick={toggleSidebar}
      className={`rounded-lg transition-all duration-200 flex items-center gap-2 ${
        isMobile 
          ? "fixed top-4 left-4 z-50 shadow-lg bg-background border border-border hover:bg-accent hover:text-accent-foreground h-10 w-10"
          : state === "expanded"
            ? "bg-destructive/10 hover:bg-destructive/20 text-destructive border border-destructive/20 hover:scale-105 px-3 py-2 h-auto"
            : "bg-primary/10 hover:bg-primary/20 text-primary border border-primary/20 hover:scale-105 px-3 py-2 h-auto"
      }`}
      title={state === "expanded" ? "Close Sidebar" : "Open Sidebar"}
    >
      <PanelLeft className={`h-5 w-5 transition-transform duration-200 ${state === "expanded" ? "rotate-180" : ""}`} />
      {!isMobile && (
        <span className="text-sm font-medium">
          {state === "expanded" ? "Close" : "Menu"}
        </span>
      )}
      <span className="sr-only">Toggle Sidebar</span>
    </Button>
  );
};

// Floating Sidebar Toggle Component for Mobile
const FloatingSidebarToggle = () => {
  const { toggleSidebar } = useSidebar();
  const isMobile = useIsMobile();

  if (!isMobile) return null;

  return (
    <Button
      onClick={toggleSidebar}
      size="icon"
      className="fixed top-4 left-4 z-50 h-10 w-10 rounded-full shadow-lg bg-background border border-border hover:bg-accent hover:text-accent-foreground"
      style={{ zIndex: 1000 }}
    >
      <PanelLeft className="h-4 w-4" />
      <span className="sr-only">Toggle Sidebar</span>
    </Button>
  );
};

const ChatInterfaceWithSidebar = () => {
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([]);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [inputMessage, setInputMessage] = useState("");
  const { toast } = useToast();
  
  // API hooks
  const askQuestionMutation = useAskQuestion();
  const healthCheck = useHealthCheck();

  // Get current chat
  const currentChat = chatSessions.find(chat => chat.id === currentChatId);
  const messages = currentChat?.messages || [];

  // Generate chat title from first message
  const generateChatTitle = (firstMessage: string) => {
    return firstMessage.length > 30 
      ? firstMessage.substring(0, 30) + "..."
      : firstMessage;
  };

  // Create new chat session
  const createNewChat = () => {
    const newChatId = Date.now().toString();
    const newChat: ChatSession = {
      id: newChatId,
      title: "নতুন চ্যাট | New Chat",
      lastMessage: "",
      timestamp: new Date(),
      messages: []
    };
    setChatSessions(prev => [newChat, ...prev]);
    setCurrentChatId(newChatId);
  };

  // Select chat
  const selectChat = (chatId: string) => {
    setCurrentChatId(chatId);
  };

  // Delete chat
  const deleteChat = (chatId: string) => {
    setChatSessions(prev => prev.filter(chat => chat.id !== chatId));
    if (currentChatId === chatId) {
      const remainingChats = chatSessions.filter(chat => chat.id !== chatId);
      setCurrentChatId(remainingChats.length > 0 ? remainingChats[0].id : null);
    }
    toast({
      title: "চ্যাট মুছে ফেলা হয়েছে | Chat Deleted",
      description: "চ্যাটটি সফলভাবে মুছে ফেলা হয়েছে | Chat has been successfully deleted",
    });
  };

  // Handle sending message
  const handleSendMessage = async () => {
    if (!inputMessage.trim()) return;

    // If no current chat, create one
    if (!currentChatId) {
      createNewChat();
      // We'll need to wait for the state to update, so we'll handle this in useEffect
      return;
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputMessage,
      isUser: true,
      timestamp: new Date(),
    };

    // Update current chat with new message
    setChatSessions(prev => prev.map(chat => {
      if (chat.id === currentChatId) {
        const updatedMessages = [...chat.messages, userMessage];
        return {
          ...chat,
          title: chat.messages.length === 0 ? generateChatTitle(inputMessage) : chat.title,
          lastMessage: inputMessage,
          timestamp: new Date(),
          messages: updatedMessages
        };
      }
      return chat;
    }));

    const currentMessage = inputMessage;
    setInputMessage("");

    try {
      // Call the API
      const response = await askQuestionMutation.mutateAsync({
        question: currentMessage,
        include_context: false
      });

      if (response.success) {
        const botMessage: Message = {
          id: (Date.now() + 1).toString(),
          text: response.answer,
          isUser: false,
          timestamp: new Date(),
        };

        setChatSessions(prev => prev.map(chat => {
          if (chat.id === currentChatId) {
            return {
              ...chat,
              lastMessage: botMessage.text,
              timestamp: new Date(),
              messages: [...chat.messages, botMessage]
            };
          }
          return chat;
        }));
      } else {
        throw new Error(response.message || 'Failed to get answer');
      }

    } catch (error) {
      console.error('API Error:', error);
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to get response from API",
        variant: "destructive",
      });
      
      // Add error message to chat
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: "দুঃখিত, আপনার প্রশ্নের উত্তর দিতে পারছি না। অনুগ্রহ করে আবার চেষ্টা করুন।",
        isUser: false,
        timestamp: new Date(),
      };

      setChatSessions(prev => prev.map(chat => {
        if (chat.id === currentChatId) {
          return {
            ...chat,
            lastMessage: errorMessage.text,
            timestamp: new Date(),
            messages: [...chat.messages, errorMessage]
          };
        }
        return chat;
      }));
    }
  };

  // Handle case where user sends message but no chat exists
  useEffect(() => {
    if (inputMessage && !currentChatId && chatSessions.length > 0) {
      const latestChat = chatSessions[0];
      setCurrentChatId(latestChat.id);
      // Trigger message send after chat is created
      setTimeout(() => handleSendMessage(), 100);
    }
  }, [chatSessions, currentChatId]);

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // Initialize with first chat if none exists
  useEffect(() => {
    if (chatSessions.length === 0) {
      createNewChat();
    }
  }, []);

  return (
    <SidebarProvider>
      <div className="min-h-screen flex w-full bg-[var(--gradient-background)] overflow-hidden sidebar-layout" style={{ minWidth: '800px' }}>
        <ChatSidebar
          chatSessions={chatSessions}
          currentChatId={currentChatId}
          onNewChat={createNewChat}
          onSelectChat={selectChat}
          onDeleteChat={deleteChat}
        />

        <div className="flex-1 flex flex-col relative min-w-0 overflow-hidden w-full main-content" style={{ minWidth: '500px', marginLeft: '0' }}>
          {/* Floating Sidebar Toggle Button for Mobile */}
          <FloatingSidebarToggle />
          {/* Header */}
          <div className="bg-card border-b border-border p-4 min-w-0 header-content" style={{ minWidth: '450px' }}>
            <div className="max-w-4xl mx-auto w-full">
              <div className="flex items-center justify-between w-full gap-4">
                <div className="flex items-center gap-3 min-w-0 flex-1 overflow-hidden header-left" style={{ minWidth: '280px' }}>
                                  <div className="p-2 flex-shrink-0">
                  <img src="/Logo_10_minute_school.png" alt="Anushilon Bot Logo" className="h-8 w-8 object-contain" />
                </div>
                  <div className="min-w-0 overflow-hidden">
                    <h1 className="text-xl md:text-2xl font-bold text-foreground truncate">10MS AI Assistant</h1>
                  </div>
                </div>
                <div className="flex items-center gap-2 md:gap-3 flex-shrink-0 header-right" style={{ minWidth: '170px' }}>
                  {/* API Status Indicator */}
                  <div className="flex items-center gap-1">
                    <div className={`w-2 h-2 rounded-full ${
                      healthCheck.isLoading 
                        ? 'bg-yellow-500 animate-pulse' 
                        : healthCheck.isError 
                          ? 'bg-red-500' 
                          : 'bg-green-500'
                    }`} />
                    <span className="text-xs text-muted-foreground hidden md:inline">
                      {healthCheck.isLoading 
                        ? 'Connecting...' 
                        : healthCheck.isError 
                          ? 'Offline' 
                          : 'Online'}
                    </span>
                  </div>
                  <ThemeToggle />
                  <CustomSidebarToggle />
                </div>
              </div>
            </div>
          </div>

          {/* Welcome Message */}
          {(!currentChat || currentChat.messages.length === 0) && (
            <div className="p-6 min-w-0">
              <div className="max-w-4xl mx-auto w-full">
                <Card className="p-4 bg-chat-background border-primary/20">
                  <div className="flex items-start gap-3">
                    <Languages className="h-5 w-5 text-primary mt-1" />
                    <div>
                      <h3 className="font-semibold text-foreground mb-2">স্বাগতম! Welcome!</h3>
                      <p className="text-sm text-muted-foreground mb-2">
                        আমি আপনাকে সাহায্য করতে পারি। আপনার প্রশ্ন ইংরেজি বা বাংলায় লিখুন।
                      </p>
                      <p className="text-sm text-muted-foreground mb-2">
                        I can help you with your questions. You can ask in English or Bengali.
                      </p>

                    </div>
                  </div>
                </Card>
              </div>
            </div>
          )}

          {/* Chat Messages */}
          <div className="flex-1 overflow-y-auto p-6 min-w-0">
            <div className="max-w-4xl mx-auto space-y-4 w-full">
              {messages.length === 0 && (
                <div className="text-center py-12">
                  <MessageCircle className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                  <p className="text-muted-foreground">আপনার প্রথম প্রশ্ন করুন | Start your first question</p>
                </div>
              )}
              
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${message.isUser ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`max-w-xs md:max-w-md lg:max-w-lg px-4 py-3 rounded-2xl ${
                      message.isUser
                        ? "bg-chat-user text-chat-user-foreground"
                        : "bg-chat-bot text-chat-bot-foreground border border-border"
                    }`}
                  >
                    <p className="text-sm leading-relaxed">{message.text}</p>
                    <p className="text-xs opacity-70 mt-1">
                      {message.timestamp.toLocaleTimeString()}
                    </p>
                  </div>
                </div>
              ))}
              
              {askQuestionMutation.isPending && (
                <div className="flex justify-start">
                  <div className="bg-chat-bot text-chat-bot-foreground border border-border px-4 py-3 rounded-2xl max-w-xs">
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" />
                      <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: "0.1s" }} />
                      <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: "0.2s" }} />
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Input Area */}
          <div className="bg-card border-t border-border p-6 min-w-0">
            <div className="max-w-4xl mx-auto w-full">
              <div className="flex gap-3 w-full">
                <Input
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="আপনার প্রশ্ন লিখুন... | Type your question..."
                  className="flex-1 py-3 min-w-0"
                  disabled={askQuestionMutation.isPending}
                />
                <Button
                  onClick={handleSendMessage}
                  disabled={!inputMessage.trim() || askQuestionMutation.isPending}
                  className="px-6 flex-shrink-0"
                >
                  <Send className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </SidebarProvider>
  );
};

export default ChatInterfaceWithSidebar;