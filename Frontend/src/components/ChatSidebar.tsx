import { useState } from "react";
import { MessageSquare, Plus, Trash2, MoreHorizontal } from "lucide-react";
import { NavLink } from "react-router-dom";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarHeader,
  useSidebar,
} from "@/components/ui/sidebar";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

interface ChatSession {
  id: string;
  title: string;
  lastMessage: string;
  timestamp: Date;
}

interface ChatSidebarProps {
  chatSessions: ChatSession[];
  currentChatId: string | null;
  onNewChat: () => void;
  onSelectChat: (chatId: string) => void;
  onDeleteChat: (chatId: string) => void;
}

export function ChatSidebar({ 
  chatSessions, 
  currentChatId, 
  onNewChat, 
  onSelectChat, 
  onDeleteChat 
}: ChatSidebarProps) {
  const { state } = useSidebar();
  const collapsed = state === "collapsed";

  const truncateText = (text: string, maxLength: number = 30) => {
    return text.length > maxLength ? text.substring(0, maxLength) + "..." : text;
  };

  return (
    <Sidebar className={`${collapsed ? "w-14" : "w-64"} flex-shrink-0 max-w-[256px]`} collapsible="offcanvas">
      <SidebarHeader className="p-4">
        <Button 
          onClick={onNewChat}
          className="w-full justify-start gap-2"
          variant="outline"
        >
          <Plus className="h-4 w-4" />
          {!collapsed && "নতুন চ্যাট | New Chat"}
        </Button>
      </SidebarHeader>

      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>
            {!collapsed && "চ্যাট ইতিহাস | Chat History"}
          </SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {chatSessions.map((chat) => (
                <SidebarMenuItem key={chat.id}>
                  <div className="flex items-center w-full group">
                    <SidebarMenuButton
                      asChild
                      className={`flex-1 ${
                        currentChatId === chat.id 
                          ? "bg-sidebar-accent text-sidebar-accent-foreground" 
                          : "hover:bg-sidebar-accent/50"
                      }`}
                    >
                      <button 
                        onClick={() => onSelectChat(chat.id)}
                        className="flex items-start gap-2 p-2 w-full text-left"
                      >
                        <MessageSquare className="h-4 w-4 mt-0.5 flex-shrink-0" />
                        {!collapsed && (
                          <div className="flex-1 min-w-0">
                            <div className="font-medium text-sm truncate">
                              {truncateText(chat.title)}
                            </div>
                            <div className="text-xs text-muted-foreground truncate">
                              {truncateText(chat.lastMessage, 25)}
                            </div>
                          </div>
                        )}
                      </button>
                    </SidebarMenuButton>
                    
                    {!collapsed && (
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-8 w-8 opacity-0 group-hover:opacity-100 transition-opacity"
                          >
                            <MoreHorizontal className="h-4 w-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuItem
                            onClick={() => onDeleteChat(chat.id)}
                            className="text-destructive focus:text-destructive"
                          >
                            <Trash2 className="h-4 w-4 mr-2" />
                            মুছে ফেলুন | Delete
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    )}
                  </div>
                </SidebarMenuItem>
              ))}
              
              {chatSessions.length === 0 && !collapsed && (
                <div className="text-center text-muted-foreground py-8 px-4">
                  <MessageSquare className="h-8 w-8 mx-auto mb-2 opacity-50" />
                  <p className="text-sm">কোনো চ্যাট নেই</p>
                  <p className="text-xs">No chats yet</p>
                </div>
              )}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
    </Sidebar>
  );
}